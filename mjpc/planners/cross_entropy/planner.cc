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

#include "mjpc/planners/cross_entropy/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <shared_mutex>

#include <absl/random/random.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;
using mjpc::spline::TimeSpline;

// initialize data and settings
void CrossEntropyPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();

  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // sampling noise
  std_initial_ =
      GetNumberOrDefault(0.1, model,
                         "sampling_exploration");        // initial variance
  std_min_ = GetNumberOrDefault(0.1, model, "std_min");  // minimum variance

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  // set number of elite samples max(best 10%, 2)
  n_elite_ =
      GetNumberOrDefault(std::max(num_trajectory_ / 10, 2), model, "n_elite");

  if (num_trajectory_ > kMaxTrajectory) {
    mju_error_i("Too many trajectories, %d is the maximum allowed.",
                kMaxTrajectory);
  }
}

// allocate memory
void CrossEntropyPlanner::Allocate() {
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

  // scratch
  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  // noise
  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  // variance
  variance.resize(model->nu * kMaxTrajectoryHorizon);  // (nu * horizon)

  // need to initialize an arbitrary order of the trajectories
  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }

  // trajectories and parameters
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
  nominal_trajectory.Initialize(num_state, model->nu, task->num_residual,
                                task->num_trace, kMaxTrajectoryHorizon);
  nominal_trajectory.Allocate(kMaxTrajectoryHorizon);
}

// reset memory to zeros
void CrossEntropyPlanner::Reset(int horizon,
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
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // variance
  double var = std_initial_ * std_initial_;
  std::fill(variance.begin(), variance.end(), var);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon);
  }
  nominal_trajectory.Reset(kMaxTrajectoryHorizon);

  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  // improvement
  improvement = 0.0;
}

// set state
void CrossEntropyPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

// optimize nominal policy using random sampling
void CrossEntropyPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  resampled_policy.plan.SetInterpolation(interpolation_);

  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;

  // n_elite_ might change in the GUI - keep constant for in this function
  n_elite_ = std::min(n_elite_, num_trajectory);
  int n_elite = std::min(n_elite_, num_trajectory);

  // resize number of mjData
  ResizeMjData(model, pool.NumThreads());

  // copy nominal policy
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    resampled_policy.CopyFrom(policy, policy.num_spline_points);
  }

  // resample nominal policy to current time
  this->ResamplePolicy(horizon);

  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();

  // simulate noisy policies
  this->Rollouts(num_trajectory, horizon, pool);

  // sort candidate policies and trajectories by score
  for (int i = 0; i < num_trajectory; i++) {
    trajectory_order[i] = i;
  }

  // sort so that the first ncandidates elements are the best candidates, and
  // the rest are in an unspecified order
  std::partial_sort(
      trajectory_order.begin(), trajectory_order.begin() + num_trajectory,
      trajectory_order.begin() + num_trajectory,
      [&trajectory = trajectory](int a, int b) {
        return trajectory[a].total_return < trajectory[b].total_return;
      });

  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = resampled_policy.num_spline_points;
  int num_parameters = num_spline_points * model->nu;

  // averaged return over elites
  double avg_return = 0.0;

  // reset parameters scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);

  // loop over elites to compute average
  for (int i = 0; i < n_elite; i++) {
    // ordered trajectory index
    int idx = trajectory_order[i];

    // add parameters
    for (int i = 0; i < num_spline_points; i++) {
      TimeSpline::Node n = candidate_policy[idx].plan.NodeAt(i);
      for (int j = 0; j < model->nu; j++) {
        parameters_scratch[i * model->nu + j] += n.values()[j];
      }
    }

    // add total return
    avg_return += trajectory[idx].total_return;
  }

  // normalize
  mju_scl(parameters_scratch.data(), parameters_scratch.data(), 1.0 / n_elite,
          num_parameters);
  avg_return /= n_elite;

  // loop over elites to compute variance
  std::fill(variance.begin(), variance.end(), 0.0);  // reset variance to zero
  for (int t = 0; t < num_spline_points; t++) {
    TimeSpline::Node n = candidate_policy[trajectory_order[0]].plan.NodeAt(t);
    for (int j = 0; j < model->nu; j++) {
      // average
      double p_avg = parameters_scratch[t * model->nu + j];
      for (int i = 0; i < n_elite; i++) {
        // candidate parameter
        double pi = n.values()[j];
        double diff = pi - p_avg;
        variance[t * model->nu + j] += diff * diff / (n_elite - 1);
      }
    }
  }

  // update
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    policy.plan.Clear();
    policy.plan.SetInterpolation(interpolation_);
    for (int t = 0; t < num_spline_points; t++) {
      absl::Span<const double> values =
          absl::MakeConstSpan(parameters_scratch.data() + t * model->nu,
                              parameters_scratch.data() + (t + 1) * model->nu);
      policy.plan.AddNode(times_scratch[t], values);
    }
  }

  // improvement: compare nominal to elite average
  improvement =
      mju_max(avg_return - trajectory[trajectory_order[0]].total_return, 0.0);

  // stop timer
  policy_update_compute_time = GetDuration(policy_update_start);
}

// compute trajectory using nominal policy
void CrossEntropyPlanner::NominalTrajectory(int horizon) {
  // set policy
  auto nominal_policy = [&cp = resampled_policy](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // rollout nominal policy
  nominal_trajectory.Rollout(nominal_policy, task, model,
                             data_[ThreadPool::WorkerId()].get(), state.data(),
                             time, mocap.data(), userdata.data(), horizon);
}
void CrossEntropyPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  NominalTrajectory(horizon);
}

// set action from policy
void CrossEntropyPlanner::ActionFromPolicy(double* action, const double* state,
                                           double time, bool use_previous) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  if (use_previous) {
    previous_policy.Action(action, state, time);
  } else {
    policy.Action(action, state, time);
  }
}

// update policy via resampling
void CrossEntropyPlanner::ResamplePolicy(int horizon) {
  // dimensions
  int num_spline_points = resampled_policy.num_spline_points;

  // time
  double nominal_time = time;
  double time_shift = mju_max(
      (horizon - 1) * model->opt.timestep / (num_spline_points - 1), 1.0e-5);

  // get spline points
  for (int t = 0; t < num_spline_points; t++) {
    times_scratch[t] = nominal_time;
    resampled_policy.Action(DataAt(parameters_scratch, t * model->nu), nullptr,
                            nominal_time);
    nominal_time += time_shift;
  }

  // copy resampled policy parameters
  resampled_policy.plan.Clear();
  for (int t = 0; t < num_spline_points; t++) {
    absl::Span<const double> values =
        absl::MakeConstSpan(parameters_scratch.data() + t * model->nu,
                            parameters_scratch.data() + (t + 1) * model->nu);
    resampled_policy.plan.AddNode(times_scratch[t], values);
  }
  resampled_policy.plan.SetInterpolation(policy.plan.Interpolation());
}

// add random noise to nominal policy
void CrossEntropyPlanner::AddNoiseToPolicy(int i, double std_min) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = candidate_policy[i].num_spline_points;
  int num_parameters = num_spline_points * model->nu;

  // sampling token
  absl::BitGen gen_;

  // shift index
  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  // sample noise
  // variance[k] is the standard deviation for the k^th control parameter over
  // the elite samples we draw a bunch of control actions from this distribution
  // (which i indexes) - the noise is stored in `noise`.
  for (int k = 0; k < num_parameters; k++) {
    noise[k + shift] = absl::Gaussian<double>(
        gen_, 0.0, std::max(std::sqrt(variance[k]), std_min));
  }

  for (int k = 0; k < candidate_policy[i].plan.Size(); k++) {
    TimeSpline::Node n = candidate_policy[i].plan.NodeAt(k);
    // add noise
    mju_addTo(n.values().data(), DataAt(noise, shift + k * model->nu),
              model->nu);
    // clamp parameters
    Clamp(n.values().data(), model->actuator_ctrlrange, model->nu);
  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void CrossEntropyPlanner::Rollouts(int num_trajectory, int horizon,
                                   ThreadPool& pool) {
  // reset noise compute time
  noise_compute_time = 0.0;

  // lock std_min
  double std_min = std_min_;

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   std_min, i]() {
      // copy nominal policy and sample noise
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        s.candidate_policy[i].CopyFrom(s.resampled_policy,
                                       s.resampled_policy.num_spline_points);
        s.candidate_policy[i].plan.SetInterpolation(
            s.resampled_policy.plan.Interpolation());

        // sample noise
        s.AddNoiseToPolicy(i, std_min);
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
  // nominal
  pool.Schedule([&s = *this, horizon]() { s.NominalTrajectory(horizon); });

  // wait
  pool.WaitCount(count_before + num_trajectory + 1);
  pool.ResetCount();
}

// returns the **nominal** trajectory (this is the purple trace)
const Trajectory* CrossEntropyPlanner::BestTrajectory() {
  return &nominal_trajectory;
}

// visualize planner-specific traces
void CrossEntropyPlanner::Traces(mjvScene* scn) {
  // sample color
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // width of a sample trace, in pixels
  double width = GetNumberOrDefault(3, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best
  auto best = this->BestTrajectory();

  // sample traces
  int n_elite = n_elite_;
  for (int k = 0; k < n_elite; k++) {
    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom + task->num_trace > scn->maxgeom) break;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                     color);

        // elite index
        int idx = trajectory_order[k];
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
void CrossEntropyPlanner::GUI(mjUI& ui) {
  mjuiDef defCrossEntropy[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &interpolation_,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      {mjITEM_SLIDERNUM, "Init. Std", 2, &std_initial_, "0 1"},
      {mjITEM_SLIDERNUM, "Min. Std", 2, &std_min_, "0.01 0.5"},
      {mjITEM_SLIDERINT, "Elite", 2, &n_elite_, "2 128"},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defCrossEntropy[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defCrossEntropy[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // set noise standard deviation limits
  mju::sprintf_arr(defCrossEntropy[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);

  // add cross entropy planner
  mjui_add(&ui, defCrossEntropy);
}

// planner-specific plots
void CrossEntropyPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                                int planner_shift, int timer_shift,
                                int planning, int* shift) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0 + planner_shift][0] + 1,
                       mju_log10(mju_max(improvement, 1.0e-6)), 100,
                       0 + planner_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Avg - Best");

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
                 1.0e-3 * policy_update_compute_time * planning, 100,
                 2 + timer_shift, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update");

  // planner shift
  shift[0] += 1;

  // timer shift
  shift[1] += 3;
}

}  // namespace mjpc
