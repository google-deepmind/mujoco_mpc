// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/planners/sample_gradient/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <shared_mutex>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/policy.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::SplineInterpolation;
using mjpc::spline::TimeSpline;

// initialize data and settings
void SampleGradientPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();

  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // exploration noise
  noise_exploration = GetNumberOrDefault(0.1, model, "sampling_exploration");

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  interpolation_ = GetNumberOrDefault(SplineInterpolation::kCubicSpline, model,
                                      "sampling_representation");

  // set number of gradient trajectories to rollout
  num_gradient_ = GetNumberOrDefault(0, model, "sample_gradient_trajectories");

  // gradient filter
  gradient_filter_ = GetNumberOrDefault(1.0, model, "sample_gradient_filter");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }
}

// allocate memory
void SampleGradientPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);
  userdata.resize(model->nuserdata);

  // policy
  int num_max_parameter = model->nu * kMaxTrajectoryHorizon;
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  resampled_policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  previous_policy.Allocate(model, *task, kMaxTrajectoryHorizon);

  // noise
  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  plan_scratch = TimeSpline(/*dim=*/model->nu);

  // need to initialize an arbitrary order of the trajectories
  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }

  // trajectories and parameters are resized and initialized in OptimizePolicy
  // need to allocate at least one for NominalTrajectory
  trajectory.resize(kMaxTrajectory);
  candidate_policy.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(state.size(), model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }

  // gradient
  gradient.resize(num_max_parameter);
  gradient_previous.resize(num_max_parameter);
}

// reset memory to zeros
void SampleGradientPlanner::Reset(int horizon,
                                  const double* initial_repeated_action) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  std::fill(userdata.begin(), userdata.end(), 0.0);
  time = 0.0;

  // policy parameters
  policy.Reset(horizon, initial_repeated_action);
  resampled_policy.Reset(horizon, initial_repeated_action);
  previous_policy.Reset(horizon, initial_repeated_action);

  // scratch
  plan_scratch.Clear();

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon);
  }

  // ctrl
  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  // improvement
  improvement = 0.0;

  // winner
  winner = 0;

  // gradient
  std::fill(gradient.begin(), gradient.end(), 0.0);
  std::fill(gradient_previous.begin(), gradient_previous.end(), 0.0);
}

// set state
void SampleGradientPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

// optimize nominal policy using random sampling and gradient search
void SampleGradientPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;

  // clamp num_gradient
  num_gradient_ = std::min(num_gradient_, num_trajectory - 1);
  int num_gradient = num_gradient_;

  // number of noisy policies
  int num_noisy = num_trajectory - num_gradient;

  // resize number of mjData
  ResizeMjData(model, pool.NumThreads());

  // copy nominal policy
  int num_spline_points = policy.num_spline_points;
  policy.plan.SetInterpolation(interpolation_);
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    resampled_policy.CopyFrom(policy, num_spline_points);
  }

  // resample nominal policy to current time
  this->ResamplePolicy(resampled_policy, horizon, num_spline_points);

  // resample gradient policies to current time
  // TODO(taylor): a bit faster to do in Rollouts, but needs more scratch to be
  // memory safe
  for (int i = 0; i < num_gradient; i++) {
    this->ResamplePolicy(candidate_policy[num_noisy + i], horizon,
                         num_spline_points);
  }

  // ----- roll out noisy policies ----- //
  // start timer
  auto perturb_rollouts_start = std::chrono::steady_clock::now();

  // roll out perturbed policies: p + s * N(0, 1)
  this->Rollouts(num_trajectory, num_gradient, horizon, pool);

  // stop timer
  rollouts_compute_time = GetDuration(perturb_rollouts_start);

  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  // initial order for partial sort
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order[i] = i;
  }

  // sort lowest to highest total return
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + num_trajectory,
      trajectory_order.begin() + num_trajectory,
      [&trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  // set winner
  if (trajectory[trajectory_order[0]].total_return <
      trajectory[idx_nominal].total_return) {
    winner = trajectory_order[0];
  } else {
    winner = idx_nominal;
  }

  // winner type
  if (winner > idx_nominal) {
    if (winner < num_trajectory - num_gradient) {
      winner_type_ = kPerturb;
    } else {
      winner_type_ = kGradient;
    }
  } else {
    winner_type_ = kNominal;
  }

  // update
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    policy.SetPlan(candidate_policy[winner].plan);
  }

  // improvement: compare nominal to winner
  improvement = mju_max(
      trajectory[idx_nominal].total_return - trajectory[winner].total_return,
      0.0);

  // stop timer
  policy_update_compute_time = GetDuration(policy_update_start);

  // ----- compute gradient candidate policies ----- //
  // start timer
  auto gradient_start = std::chrono::steady_clock::now();

  // candidate policies
  this->GradientCandidates(num_trajectory, num_gradient, horizon, pool);

  // stop timer
  gradient_candidates_compute_time = GetDuration(gradient_start);
}

// compute trajectory using nominal policy
void SampleGradientPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // set policy
  auto nominal_policy = [&cp = resampled_policy](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // rollout nominal policy
  trajectory[idx_nominal].Rollout(nominal_policy, task, model, data_[0].get(),
                                  state.data(), time, mocap.data(),
                                  userdata.data(), horizon);
}

// set action from policy
void SampleGradientPlanner::ActionFromPolicy(double* action,
                                             const double* state, double time,
                                             bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

// update policy via resampling
void SampleGradientPlanner::ResamplePolicy(
    SamplingPolicy& policy, int horizon, int num_spline_points) {
  // dimension

  // time
  double nominal_time = time;
  double time_shift = mju_max(
      (horizon - 1) * model->opt.timestep / (num_spline_points - 1), 1.0e-5);

  // get spline points
  plan_scratch.Clear();
  plan_scratch.Reserve(num_spline_points);
  plan_scratch.SetInterpolation(policy.plan.Interpolation());
  for (int t = 0; t < num_spline_points; t++) {
    TimeSpline::Node node = plan_scratch.AddNode(nominal_time);
    policy.Action(node.values().data(), /*state=*/nullptr, nominal_time);
    nominal_time += time_shift;
  }

  // copy resampled policy parameters
  policy.SetPlan(plan_scratch);

  // set dimensions
  policy.num_spline_points = num_spline_points;
}

// add random noise to nominal policy
void SampleGradientPlanner::AddNoiseToPolicy(int i) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = candidate_policy[i].num_spline_points;
  // sampling token
  absl::BitGen gen_;

  // shift index
  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  // sample noise
  for (int k = 0; k < num_spline_points * model->nu; k++) {
    noise[k + shift] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  for (int j = 0; j < num_spline_points; j++) {
    TimeSpline::Node node = candidate_policy[i].plan.NodeAt(j);
    mju_addToScl(node.values().data(), DataAt(noise, j * model->nu + shift),
                 noise_exploration, model->nu);
    Clamp(node.values().data(), model->actuator_ctrlrange, model->nu);
  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// rollout candidate policies
void SampleGradientPlanner::Rollouts(int num_trajectory, int num_gradient,
                                     int horizon, ThreadPool& pool) {
  // reset perturbation compute time
  noise_compute_time = 0.0;

  // search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   idx_nominal = this->idx_nominal, num_trajectory,
                   num_gradient, i]() {
      // nominal and noisy policies
      if (i < num_trajectory - num_gradient) {
        // copy nominal policy
        s.candidate_policy[i].CopyFrom(s.resampled_policy,
                                       s.resampled_policy.num_spline_points);

        // noisy nominal policy
        if (i > idx_nominal) s.AddNoiseToPolicy(i);
      }

      // ----- rollout sample policy ----- //

      // policy
      auto sample_policy_i = [&candidate_policy = s.candidate_policy, &i](
                                 double* action, const double* state,
                                 double time) {
        candidate_policy[i].Action(action, state, time);
      };

      // policy rollout
      s.trajectory[i].Rollout(
          sample_policy_i, task, model, s.data_[ThreadPool::WorkerId()].get(),
          state.data(), time, mocap.data(), userdata.data(), horizon);
    });
  }
  pool.WaitCount(count_before + num_trajectory);
  pool.ResetCount();
}

// compute candidate trajectories
void SampleGradientPlanner::GradientCandidates(int num_trajectory,
                                               int num_gradient, int horizon,
                                               ThreadPool& pool) {
  if (num_gradient < 1) return;

  // number of parameters
  int num_spline_points = resampled_policy.num_spline_points;
  int num_parameters = num_spline_points * model->nu;

  // cache old gradient
  mju_copy(gradient_previous.data(), gradient.data(), num_parameters);

  // -- compute approximate gradient -- //
  // average return
  int num_noisy = num_trajectory - num_gradient;

  // fitness shaping
  // https://www.jmlr.org/papers/volume15/wierstra14a/wierstra14a.pdf
  if (return_weight_.size() != num_noisy) {
    // resize number of weights
    return_weight_.resize(num_noisy);

    // -- sort noisy samples only (exclude gradient samples) -- //
    // initial order for partial sort
    for (int i = 0; i < num_noisy; i++) {
      trajectory_order[i] = i;
    }

    // sort lowest to highest total return
    std::partial_sort(
        trajectory_order.begin(), trajectory_order.begin() + num_noisy,
        trajectory_order.begin() + num_noisy,
        [&trajectory = trajectory](int a, int b) {
          return trajectory[a].total_return < trajectory[b].total_return;
        });

    // compute normalization
    double f0 = std::log(0.5 * num_noisy + 1.0);
    double den = 0.0;
    for (int i = 0; i < num_noisy; i++) {
      den += std::max(0.0, f0 - std::log(trajectory_order[i] + 1));
    }

    // compute weights
    for (int i = 0; i < num_noisy; i++) {
      return_weight_[i] =
          std::max(0.0, f0 - std::log(trajectory_order[i] + 1)) / den -
          1.0 / num_noisy;
    }
  }

  // gradient
  std::fill(gradient.begin(), gradient.end(), 0.0);
  for (int i = 0; i < num_noisy; i++) {
    double* noisei = noise.data() +
                     trajectory_order[i] * (model->nu * kMaxTrajectoryHorizon);
    mju_addToScl(gradient.data(), noisei, return_weight_[i] / num_noisy,
                 num_parameters);
  }

  // compute step sizes for gradient direction
  if (step_size_.size() != num_gradient) {
    step_size_.resize(num_gradient);
    LogScale(step_size_.data(), gradient_max_step_size, gradient_min_step_size,
             num_gradient);
  }

  // gradient filter gf * grad + (1 - gf) * grad_prev
  double gradient_filter = gradient_filter_;

  // compute candidate policies along gradient direction
  // these candidates will be evaluated at the next planning iteration
  for (int i = num_noisy; i < num_trajectory; i++) {
    // copy nominal policy
    candidate_policy[i].CopyFrom(resampled_policy, num_spline_points);

    // scaling
    double scaling = step_size_[i - num_noisy] / noise_exploration;

    // gradient step
    for (int t = 0; t < candidate_policy[i].plan.Size(); t++) {
      TimeSpline::Node n = candidate_policy[i].plan.NodeAt(t);
      mju_addToScl(n.values().data(), gradient.data() + t * model->nu,
                   -scaling * gradient_filter, model->nu);

      // TODO(taylor): resample the gradient_previous?
      mju_addToScl(n.values().data(), gradient_previous.data() + t * model->nu,
                   -scaling * (1.0 - gradient_filter), model->nu);
      // clamp parameters
      Clamp(n.values().data(), model->actuator_ctrlrange, model->nu);
    }
  }
}

// returns the nominal trajectory (this is the purple trace)
const Trajectory* SampleGradientPlanner::BestTrajectory() {
  return &trajectory[winner];
}

// visualize planner-specific traces
void SampleGradientPlanner::Traces(mjvScene* scn) {
  // noisy sample: white
  float white[4];
  white[0] = 1.0;
  white[1] = 1.0;
  white[2] = 1.0;
  white[3] = 1.0;

  // gradient sample: orange
  float orange[4];
  orange[0] = 1.0;
  orange[1] = 0.5;
  orange[2] = 0.0;
  orange[3] = 1.0;

  // width of a sample trace, in pixels
  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best
  auto best = this->BestTrajectory();

  // check sizes
  int num_trajectory = num_trajectory_;
  int num_gradient = num_gradient_;
  int num_noisy = num_trajectory - num_gradient;

  // traces between Newton and Cauchy points
  for (int k = 1; k < num_trajectory; k++) {
    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // index
        int idx = trajectory_order[k];

        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     idx < num_noisy ? white : orange);

        // make geometry
        mjv_makeConnector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectory[idx].trace[3 * task->num_trace * i + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * i + 1 + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * i + 2 + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * (i + 1) + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * (i + 1) + 1 + 3 * j],
            trajectory[idx].trace[3 * task->num_trace * (i + 1) + 2 + 3 * j]);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void SampleGradientPlanner::GUI(mjUI& ui) {
  mjuiDef defSampleGradient[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_, "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Noise Std.", 2, &noise_exploration, "0 1"},
      {mjITEM_SLIDERINT, "Grad. Rollouts", 2, &num_gradient_, "0 1"},
      {mjITEM_SLIDERNUM, "Grad. Filter", 2, &gradient_filter_, "0 1"},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampleGradient[0].other, "%i %i", 2, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampleGradient[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampleGradient[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);

  // set number of gradient trajectory slider limits
  mju::sprintf_arr(defSampleGradient[4].other, "%i %i", 0, kMaxTrajectory);

  // add sample gradient planner
  mjui_add(&ui, defSampleGradient);
}

// planner-specific plots
void SampleGradientPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                                  int planner_shift, int timer_shift,
                                  int planning, int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  // winner plot value
  double winner_plot_val = 0.0;  // nominal
  if (winner_type_ == kPerturb) {
    winner_plot_val = -6.0;
  } else if (winner_type_ == kGradient) {
    int num_noisy = num_trajectory_ - num_gradient_;
    winner_plot_val = 6.0 * (winner - num_noisy) / num_gradient_;
  }

  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[1 + planner_shift][0] + 1,
                       winner_plot_val, 100, 1 + planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improvement");
  mju::strcpy_arr(fig_planner->linename[1 + planner_shift],
                  "Perturb|Nominal|Gradient");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // bounds
  double timer_bounds[2] = {0.0, 1.0};

  // ----- timer ----- //

  PlotUpdateData(
      fig_timer, timer_bounds, fig_timer->linedata[0 + timer_shift][0] + 1,
      1.0e-3 * noise_compute_time * planning, 100, 0 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[1 + timer_shift][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100,
                 1 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[2 + timer_shift][0] + 1,
                 1.0e-3 * gradient_candidates_compute_time * planning, 100,
                 2 + timer_shift, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds,
                 fig_timer->linedata[3 + timer_shift][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100,
                 3 + timer_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollouts");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Gradient Candidates");
  mju::strcpy_arr(fig_timer->linename[3 + timer_shift], "Policy Update");

  // planner shift
  shift[0] += 2;

  // timer shift
  shift[1] += 4;
}

}  // namespace mjpc
