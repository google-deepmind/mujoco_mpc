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

#include "planners/sampling/planner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <mutex>
#include <shared_mutex>

#include <absl/random/random.h>
#include "array_safety.h"
#include "planners/sampling/policy.h"
#include "states/state.h"
#include "trajectory.h"
#include "utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;

// initialize data and settings
void SamplingPlanner::Initialize(mjModel* model, const Task& task) {
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

  // sampling noise
  noise_exploration = GetNumberOrDefault(0.1, model, "sampling_exploration");

  // set number of trajectories to rollout
  num_trajectories_ = GetNumberOrDefault(10, model, "sampling_trajectories");

  winner = 0;
}

// allocate memory
void SamplingPlanner::Allocate() {
  // initial state
  int num_state = model->nq + model->nv + model->na;

  // state
  state.resize(num_state);
  mocap.resize(7 * model->nmocap);

  // policy
  int num_max_parameter = model->nu * kMaxTrajectoryHorizon;
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);

  // scratch
  parameters_scratch.resize(num_max_parameter);
  times_scratch.resize(kMaxTrajectoryHorizon);

  // noise
  noise.resize(kMaxTrajectory * (model->nu * kMaxTrajectoryHorizon));

  // trajectory and parameters
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectories[i].Initialize(num_state, model->nu, task->num_residual,
                               task->num_trace, kMaxTrajectoryHorizon);
    trajectories[i].Allocate(kMaxTrajectoryHorizon);
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }

  // noise gradient
  noise_gradient.resize(num_max_parameter);
}

// reset memory to zeros
void SamplingPlanner::Reset(int horizon) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 7 * model->nmocap);
  time = 0.0;

  // policy parameters
  policy.Reset(horizon);

  // scratch
  std::fill(parameters_scratch.begin(), parameters_scratch.end(), 0.0);
  std::fill(times_scratch.begin(), times_scratch.end(), 0.0);

  // noise
  std::fill(noise.begin(), noise.end(), 0.0);

  // trajectory samples
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectories[i].Reset(kMaxTrajectoryHorizon);
    candidate_policy[i].Reset(horizon);
  }

  for (const auto& d : data_) {
    mju_zero(d->ctrl, model->nu);
  }

  // noise gradient
  std::fill(noise_gradient.begin(), noise_gradient.end(), 0.0);

  // improvement
  improvement = 0.0;

  // winner
  winner = 0;
}

// set state
void SamplingPlanner::SetState(State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), &this->time);
}

// optimize nominal policy using random sampling
void SamplingPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // if num_trajectories_ has changed, use it in this new iteration.
  // num_trajectories_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectories = num_trajectories_;
  ResizeMjData(model, pool.NumThreads());

  // timers
  double nominal_time = 0.0;
  double rollouts_time = 0.0;
  double policy_update_time = 0.0;

  // ----- nominal policy ----- //
  // start timer
  auto nominal_start = std::chrono::steady_clock::now();

  // copy nominal policy
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    policy.num_parameters = model->nu * policy.num_spline_points;  // set
    candidate_policy[0].CopyFrom(policy, policy.num_spline_points);
  }

  // resample policy
  this->ResamplePolicy(horizon);

  // stop timer
  nominal_time += std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - nominal_start)
                      .count();

  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();

  // sample random policies and rollout
  double best_return = trajectories[winner].total_return;

  // simulate noisy policies
  this->Rollouts(num_trajectories, horizon, pool);

  // ----- compare rollouts ----- //
  // reset
  winner = 0;

  // random search
  for (int i = 1; i < num_trajectories; i++) {
    if (trajectories[i].total_return < trajectories[winner].total_return) {
      winner = i;
    }
  }

  // stop timer
  rollouts_time += std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - rollouts_start)
                       .count();

  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  // copy best candidate policy
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    policy.CopyParametersFrom(candidate_policy[winner].parameters,
                    candidate_policy[winner].times);
  }

  // stop timer
  policy_update_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - policy_update_start)
          .count();

  // improvement
  improvement = mju_max(best_return - trajectories[winner].total_return, 0.0);

  // set timers
  nominal_compute_time = nominal_time;
  rollouts_compute_time = rollouts_time;
  policy_update_compute_time = policy_update_time;
}

// compute trajectory using nominal policy
void SamplingPlanner::NominalTrajectory(int horizon) {
  // set policy
  auto nominal_policy = [&cp = candidate_policy[0]](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // rollout nominal policy
  trajectories[0].Rollout(nominal_policy, task, model, data_[0].get(),
                          state.data(), time, mocap.data(), horizon);
}

// set action from policy
void SamplingPlanner::ActionFromPolicy(double* action, const double* state,
                                       double time) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  policy.Action(action, state, time);
}

// update policy via resampling
void SamplingPlanner::ResamplePolicy(int horizon) {
  // dimensions
  int num_spline_points = candidate_policy[0].num_spline_points;

  // set time
  double nominal_time = time;
  double time_shift = mju_max(
      (horizon - 1) * model->opt.timestep / (num_spline_points - 1),
      1.0e-5);

  // get spline points
  for (int t = 0; t < num_spline_points; t++) {
    times_scratch[t] = nominal_time;
    candidate_policy[0].Action(DataAt(parameters_scratch, t * model->nu),
                               nullptr, nominal_time);
    nominal_time += time_shift;
  }

  // copy policy parameters
  candidate_policy[0].CopyParametersFrom(parameters_scratch, times_scratch);

  // time power transformation
  PowerSequence(candidate_policy[0].times.data(), time_shift,
                candidate_policy[0].times[0],
                candidate_policy[0].times[num_spline_points - 1],
                timestep_power, num_spline_points);
}

// add random noise to nominal policy
void SamplingPlanner::AddNoiseToPolicy(int i) {
  // start timer
  auto noise_start = std::chrono::steady_clock::now();

  // dimensions
  int num_spline_points = candidate_policy->num_spline_points;
  int num_parameters = candidate_policy->num_parameters;

  // sampling token
  absl::BitGen gen_;

  // shift index
  int shift = i * (model->nu * kMaxTrajectoryHorizon);

  // sample noise
  for (int k = 0; k < num_parameters; k++) {
    noise[k + shift] = absl::Gaussian<double>(gen_, 0.0, noise_exploration);
  }

  // copy policy
  candidate_policy[i].CopyFrom(candidate_policy[0], num_spline_points);

  // add noise
  mju_addTo(candidate_policy[i].parameters.data(), DataAt(noise, shift),
            num_parameters);

  // clamp parameters
  for (int t = 0; t < num_spline_points; t++) {
    Clamp(DataAt(candidate_policy[i].parameters, t * model->nu),
          model->actuator_ctrlrange, model->nu);
  }

  // end timer
  IncrementAtomic(noise_compute_time,
                  std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - noise_start)
                      .count());
}

// compute candidate trajectories
void SamplingPlanner::Rollouts(int num_trajectories, int horizon,
                               ThreadPool& pool) {
  // reset noise compute time
  noise_compute_time = 0.0;

  // random search
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectories; i++) {
    pool.Schedule([&s = *this, &model = this->model, &task = this->task,
                   &state = this->state, &time = this->time,
                   &mocap = this->mocap, horizon, i]() {
      // sample noise policy
      if (i != 0) s.AddNoiseToPolicy(i);

      // set policy representation
      s.candidate_policy[i].representation = s.policy.representation;

      // ----- rollout sample policy ----- //

      // policy
      auto sample_policy_i = [&candidate_policy = s.candidate_policy, &i](
                                 double* action, const double* state,
                                 double time) {
        candidate_policy[i].Action(action, state, time);
      };

      // policy rollout
      s.trajectories[i].Rollout(sample_policy_i, task, model,
                                s.data_[ThreadPool::WorkerId()].get(),
                                state.data(), time, mocap.data(), horizon);
    });
  }
  pool.WaitCount(count_before + num_trajectories);
  pool.ResetCount();
}

// return trajectory with best total return
const Trajectory* SamplingPlanner::BestTrajectory() {
  return &trajectories[winner];
}

// visualize planner-specific traces
void SamplingPlanner::Traces(mjvScene* scn) {
  // sample color
  float color[4];
  color[0] = 1.0;
  color[1] = 1.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // sample width
  double width = GetNumberOrDefault(0.01, model, "agent_sample_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // best
  auto best = this->BestTrajectory();

  // sample traces
  for (int k = 0; k < num_trajectories_; k++) {
    // skip winner
    if (k == winner) continue;

    // plot sample
    for (int i = 0; i < best->horizon - 1; i++) {
      if (scn->ngeom >= scn->maxgeom) continue;
      for (int j = 0; j < task->num_trace; j++) {
        // initialize geometry
        mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_LINE, zero3, zero3, zero9,
                    color);

        // make geometry
        mjv_makeConnector(
            &scn->geoms[scn->ngeom], mjGEOM_LINE, width,
            trajectories[k].trace[3 * task->num_trace * i + 3 * j],
            trajectories[k].trace[3 * task->num_trace * i + 1 + 3 * j],
            trajectories[k].trace[3 * task->num_trace * i + 2 + 3 * j],
            trajectories[k].trace[3 * task->num_trace * (i + 1) + 3 * j],
            trajectories[k].trace[3 * task->num_trace * (i + 1) + 1 + 3 * j],
            trajectories[k].trace[3 * task->num_trace * (i + 1) + 2 + 3 * j]);

        // increment number of geometries
        scn->ngeom += 1;
      }
    }
  }
}

// planner-specific GUI elements
void SamplingPlanner::GUI(mjUI& ui) {
  mjuiDef defSampling[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectories_, "0 1"},
      {mjITEM_SELECT, "Spline", 2, &policy.representation,
       "Zero\nLinear\nCubic"},
      {mjITEM_SLIDERINT, "Spline Pts", 2, &policy.num_spline_points, "0 1"},
      // {mjITEM_SLIDERNUM, "Spline Pow. ", 2, &timestep_power, "0 10"},
      // {mjITEM_SELECT, "Noise type", 2, &noise_type, "Gaussian\nUniform"},
      {mjITEM_SLIDERNUM, "Noise Std", 2, &noise_exploration, "0 1"},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defSampling[0].other, "%i %i", 1, kMaxTrajectory);

  // set spline point limits
  mju::sprintf_arr(defSampling[2].other, "%i %i", MinSamplingSplinePoints,
                   MaxSamplingSplinePoints);

  // set noise standard deviation limits
  mju::sprintf_arr(defSampling[3].other, "%f %f", MinNoiseStdDev,
                   MaxNoiseStdDev);

  // add sampling planner
  mjui_add(&ui, defSampling);
}

// planner-specific plots
void SamplingPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                            int planning) {
  // ----- planner ----- //
  double planner_bounds[2] = {-6.0, 6.0};

  // improvement
  mjpc::PlotUpdateData(
      fig_planner, planner_bounds, fig_planner->linedata[0][0] + 1,
      mju_log10(mju_max(improvement, 1.0e-6)), 100, 0, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0], "Improvement");

  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // bounds
  double timer_bounds[2] = {0.0, 1.0};

  // ----- timer ----- //

  // update plots
  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[7][0] + 1,
                 1.0e-3 * nominal_compute_time * planning, 100, 7, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[10][0] + 1,
                 1.0e-3 * noise_compute_time * planning, 100, 10, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[1][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100, 1, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[9][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100, 9, 0, 1,
                 -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[7], "Nominal");
  mju::strcpy_arr(fig_timer->linename[10], "Noise");
  mju::strcpy_arr(fig_timer->linename[1], "Rollout");
  mju::strcpy_arr(fig_timer->linename[9], "Policy Update");
}

}  // namespace mjpc
