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

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <shared_mutex>

#include "mjpc/array_safety.h"
#include "mjpc/planners/policy.h"
#include "mjpc/states/state.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;

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

  // rollout parameters
  timestep_power = 1.0;

  // sampling perturbation
  scale = GetNumberOrDefault(0.1, model,
                             "sampling_exploration");

  // set number of trajectories to rollout
  num_trajectory_ = GetNumberOrDefault(10, model, "sampling_trajectories");

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

  // scratch
  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  // perturbation
  perturbation.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  // need to initialize an arbitrary order of the trajectories
  trajectory_order.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory_order[i] = i;
  }

  // trajectories and parameters
  trajectory.resize(kMaxTrajectory);
  candidate_policy.resize(kMaxTrajectory);
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(num_state, model->nu, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }
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
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // perturbation
  std::fill(perturbation.begin(), perturbation.end(), 0.0);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon);
  }

  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  // improvement
  improvement = 0.0;

  // winner
  winner = 0;
}

// set state
void SampleGradientPlanner::SetState(const State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), this->userdata.data(),
               &this->time);
}

// optimize nominal policy using random sampling
void SampleGradientPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = num_trajectory_;

  // resize number of mjData
  ResizeMjData(model, pool.NumThreads());

  // copy nominal policy
  policy.num_parameters = model->nu * policy.num_spline_points;
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

  // set winner
  winner = 0;
  if (trajectory[trajectory_order[0]].total_return <
      trajectory[0].total_return) {
    winner = trajectory_order[0];
  }

  // stop timer
  rollouts_compute_time = GetDuration(rollouts_start);

  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  // update
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    policy.CopyParametersFrom(candidate_policy[winner].parameters,
                              times_scratch);
  }

  // improvement: compare nominal to elite average
  improvement = mju_max(
      trajectory[0].total_return - trajectory[winner].total_return, 0.0);

  // stop timer
  policy_update_compute_time = GetDuration(policy_update_start);
}

// compute trajectory using nominal policy
void SampleGradientPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  // set policy
  auto nominal_policy = [&cp = resampled_policy](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // rollout nominal policy
  trajectory[0].Rollout(nominal_policy, task, model, data_[0].get(),
                        state.data(), time, mocap.data(), userdata.data(),
                        horizon);
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
void SampleGradientPlanner::ResamplePolicy(int horizon) {
  // dimensions
  int num_parameters = resampled_policy.num_parameters;
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
  mju_copy(resampled_policy.parameters.data(), parameters_scratch.data(),
           num_parameters);
  mju_copy(resampled_policy.times.data(), times_scratch.data(),
           num_spline_points);

  // time step power scaling
  PowerSequence(resampled_policy.times.data(), time_shift,
                resampled_policy.times[0],
                resampled_policy.times[num_spline_points - 1], timestep_power,
                num_spline_points);
}

// add random perturbation to nominal policy
void SampleGradientPlanner::AddNoiseToPolicy(int i) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = candidate_policy[i].num_spline_points;
  int num_parameters = candidate_policy[i].num_parameters;

  // sampling token
  absl::BitGen gen_;

  // shift index
  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  // sample perturbation
  for (int k = 0; k < num_parameters; k++) {
    perturbation[k + shift] = absl::Gaussian<double>(gen_, 0.0, scale);
  }

  // add perturbation
  mju_addTo(candidate_policy[i].parameters.data(), DataAt(perturbation, shift),
            num_parameters);

  // clamp parameters
  for (int t = 0; t < num_spline_points; t++) {
    Clamp(DataAt(candidate_policy[i].parameters, t * model->nu),
          model->actuator_ctrlrange, model->nu);
  }

  // end timer
  IncrementAtomic(noise_compute_time, GetDuration(noise_start));
}

// compute candidate trajectories
void SampleGradientPlanner::Rollouts(int num_trajectory, int horizon,
                                     ThreadPool& pool) {
  // reset perturbation compute time
  noise_compute_time = 0.0;

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, &userdata = this->userdata, horizon,
                   i]() {
      // copy nominal policy and sample perturbation
      {
        const std::shared_lock<std::shared_mutex> lock(s.mtx_);
        s.candidate_policy[i].CopyFrom(s.resampled_policy,
                                       s.resampled_policy.num_spline_points);
        s.candidate_policy[i].representation =
            s.resampled_policy.representation;

        // sample perturbation
        if (i != 0) s.AddNoiseToPolicy(i);
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

// returns the nominal trajectory (this is the purple trace)
const Trajectory* SampleGradientPlanner::BestTrajectory() {
  return &trajectory[winner];
}

// visualize planner-specific traces
void SampleGradientPlanner::Traces(mjvScene* scn) {
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
  int num_trajectory = num_trajectory_;
  for (int k = 0; k < num_trajectory; k++) {
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
void SampleGradientPlanner::GUI(mjUI& ui) {
  mjuiDef defSampleGradient[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &policy.representation,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      // {mjITEM_SLIDERNUM, "Spline Pow. ", 2, &timestep_power, "0 10"},
      // {mjITEM_SELECT, "Noise type", 2, &noise_type, "Gaussian\nUniform"},
      {mjITEM_SLIDERNUM, "Scale", 2, &scale, "0 1"},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampleGradient[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampleGradient[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // set perturbation standard deviation limits
  mju::sprintf_arr(defSampleGradient[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);

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

  // legend
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improvement");

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
