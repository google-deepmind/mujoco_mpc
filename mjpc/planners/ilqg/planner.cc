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

#include "planners/ilqg/planner.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>

#include "array_safety.h"
#include "planners/ilqg/backward_pass.h"
#include "planners/ilqg/policy.h"
#include "planners/ilqg/settings.h"
#include "planners/planner.h"
#include "states/state.h"
#include "trajectory.h"
#include "utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// initialize data and settings
void iLQGPlanner::Initialize(mjModel* model, const Task& task) {
  // delete mjData instances since model might have changed.
  data_.clear();

  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  // model
  this->model = model;

  // task
  this->task = &task;

  // dimensions
  dim_state = model->nq + model->nv + model->na;  // state dimension
  dim_state_derivative =
      2 * model->nv + model->na;    // state derivative dimension
  dim_action = model->nu;           // action dimension
  dim_sensor = model->nsensordata;  // number of sensor values
  dim_max = 10 * mju_max(mju_max(mju_max(dim_state, dim_state_derivative),
                                 dim_action),
                         model->nuser_sensor);
  num_trajectory = GetNumberOrDefault(10, model, "ilqg_num_rollouts");
  settings.regularization_type = GetNumberOrDefault(
      settings.regularization_type, model, "ilqg_regularization_type");
}

// allocate memory
void iLQGPlanner::Allocate() {
  // state
  state.resize(model->nq + model->nv + model->na);
  mocap.resize(7 * model->nmocap);

  // candidate trajectories
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Initialize(dim_state, dim_action, task->num_residual,
                             task->num_trace, kMaxTrajectoryHorizon);
    trajectory[i].Allocate(kMaxTrajectoryHorizon);
  }

  // model derivatives
  model_derivative.Allocate(dim_state_derivative, dim_action, dim_sensor,
                            kMaxTrajectoryHorizon);

  // costs derivatives
  cost_derivative.Allocate(dim_state_derivative, dim_action, task->num_residual,
                           kMaxTrajectoryHorizon, dim_max);

  // backward pass
  backward_pass.Allocate(dim_state_derivative, dim_action,
                         kMaxTrajectoryHorizon);

  // policy
  policy.Allocate(model, *task, kMaxTrajectoryHorizon);
  for (int i = 0; i < kMaxTrajectory; i++) {
    candidate_policy[i].Allocate(model, *task, kMaxTrajectoryHorizon);
  }

  // ----- boxQP ----- //
  boxqp.Allocate(dim_action);
}

// reset memory to zeros
void iLQGPlanner::Reset(int horizon) {
  // state
  std::fill(state.begin(), state.end(), 0.0);
  std::fill(mocap.begin(), mocap.end(), 0.0);
  time = 0.0;

  // model derivatives
  model_derivative.Reset(dim_state_derivative, dim_action, dim_sensor, horizon);

  // cost derivatives
  cost_derivative.Reset(dim_state_derivative, dim_action, task->num_residual,
                        horizon);

  // backward pass
  backward_pass.Reset(dim_state_derivative, dim_action, horizon);

  // policy
  policy.Reset(horizon);
  for (int i = 0; i < kMaxTrajectory; i++) {
    candidate_policy[i].Reset(horizon);
  }

  // candidate trajectories
  for (int i = 0; i < kMaxTrajectory; i++) {
    trajectory[i].Reset(horizon);
  }

  // values
  step_size = 0.0;
  improvement = 0.0;
  expected = 0.0;
  surprise = 0.0;
}

// set state
void iLQGPlanner::SetState(State& state) {
  state.CopyTo(this->state.data(), this->mocap.data(), &this->time);
}

// optimize nominal policy using iLQG
void iLQGPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  ResizeMjData(model, pool.NumThreads());

  // timers
  double nominal_time = 0.0;
  double model_derivative_time = 0.0;
  double cost_derivative_time = 0.0;
  double rollouts_time = 0.0;
  double backward_pass_time = 0.0;
  double policy_update_time = 0.0;

  // maximum number of trajectories in linesearch
  num_trajectory = mju_min(num_trajectory, kMaxTrajectory);

  // ----- nominal rollout ----- //
  // start timer
  auto nominal_start = std::chrono::steady_clock::now();

  // copy nominal policy
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    candidate_policy[0].CopyFrom(policy, horizon);
    candidate_policy[0].representation = policy.representation;
  }

  // rollout nominal policy
  this->NominalTrajectory(horizon);
  if (trajectory[0].failure) {
    std::cerr << "Nominal trajectory diverged.\n";
  }

  // set previous best cost
  double c_prev = trajectory[0].total_return;

  // set candidate policy nominal trajectory
  candidate_policy[0].trajectory = trajectory[0];

  // end timer
  nominal_time = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - nominal_start)
                     .count();

  // rollouts
  double c_best = c_prev;
  for (int i = 0; i < settings.max_rollout; i++) {
    // ----- model derivatives ----- //
    // start timer
    auto model_derivative_start = std::chrono::steady_clock::now();

    // compute model and sensor Jacobians
    model_derivative.Compute(
        model, data_, candidate_policy[0].trajectory.states.data(),
        candidate_policy[0].trajectory.actions.data(),
        candidate_policy[0].trajectory.times.data(), dim_state,
        dim_state_derivative, dim_action, dim_sensor, horizon,
        settings.fd_tolerance, settings.fd_mode, pool);

    // stop timer
    model_derivative_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - model_derivative_start)
            .count();

    // ----- cost derivatives ----- //
    // start timer
    auto cost_derivative_start = std::chrono::steady_clock::now();

    // cost derivatives
    cost_derivative.Compute(
        candidate_policy[0].trajectory.residual.data(),
        model_derivative.C.data(), model_derivative.D.data(),
        dim_state_derivative, dim_action, dim_max, dim_sensor,
        task->num_residual, task->dim_norm_residual.data(), task->num_cost,
        task->weight.data(), task->norm.data(), task->num_parameter.data(),
        task->num_norm_parameter.data(), task->risk, horizon, pool);

    // end timer
    cost_derivative_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - cost_derivative_start)
            .count();

    // ----- backward pass ----- //
    // start timer
    auto backward_pass_start = std::chrono::steady_clock::now();

    // compute feedback gains and action improvement via Riccati
    backward_pass.Riccati(&candidate_policy[0], &model_derivative,
                          &cost_derivative, dim_state_derivative, dim_action,
                          horizon, backward_pass.regularization, boxqp,
                          candidate_policy[0].trajectory.actions.data(),
                          model->actuator_ctrlrange, settings);

    // end timer
    backward_pass_time +=
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now() - backward_pass_start)
            .count();

    // ----- rollout policy ----- //
    auto rollouts_start = std::chrono::steady_clock::now();

    // copy policy
    for (int j = 1; j < num_trajectory; j++) {
      candidate_policy[j].CopyFrom(candidate_policy[0], horizon);
      candidate_policy[j].representation = candidate_policy[0].representation;
    }

    // improvement step sizes (log scaling)
    LogScale(improvement_step, 1.0, settings.min_step_size, num_trajectory - 1);
    improvement_step[num_trajectory - 1] = 0.0;

    // feedback rollouts (parallel)
    this->Rollouts(horizon, pool);

    // ----- evaluate rollouts ------ //
    winner = num_trajectory - 1;
    int failed = 0;
    if (trajectory[num_trajectory - 1].failure) {
      failed++;
    }
    for (int j = num_trajectory - 2; j >= 0; j--) {
      if (trajectory[j].failure) {
        failed++;
        continue;
      }
      // compute cost
      double c_sample = trajectory[j].total_return;

      // compare cost
      if (c_sample < c_best) {
        c_best = c_sample;
        winner = j;
      }
    }
    if (failed) {
      std::cerr << "iLQG: " << failed << " out of " << num_trajectory
                << " rollouts failed.\n";
    }

    // update nominal with winner
    candidate_policy[0].trajectory = trajectory[winner];

    // improvement
    step_size = improvement_step[winner];
    expected = -1.0 * step_size *
                   (backward_pass.dV[0] + step_size * backward_pass.dV[1]) +
               1.0e-16;
    improvement = c_prev - c_best;
    surprise = mju_min(mju_max(0, improvement / expected), 2);

    // update regularization
    backward_pass.UpdateRegularization(settings.min_regularization,
                                       settings.max_regularization, surprise,
                                       step_size);

    if (settings.verbose) {
      std::cout << "dV: " << expected << '\n';
      std::cout << "dV[0]: " << backward_pass.dV[0] << '\n';
      std::cout << "dV[1]: " << backward_pass.dV[1] << '\n';
      std::cout << "c_best: " << c_best << '\n';
      std::cout << "c_prev: " << c_prev << '\n';
      std::cout << "c_nominal: " << policy.trajectory.total_return << '\n';
      std::cout << "step size: " << step_size << '\n';
      std::cout << "improvement: " << improvement << '\n';
      std::cout << "regularization: " << backward_pass.regularization << '\n';
      std::cout << "factor: " << backward_pass.regularization_factor << '\n';
      std::cout << std::endl;
    }

    // stop timer
    rollouts_time += std::chrono::duration_cast<std::chrono::microseconds>(
                         std::chrono::steady_clock::now() - rollouts_start)
                         .count();
  }

  // ----- policy update ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();
  {
    const std::shared_lock<std::shared_mutex> lock(mtx_);
    // improvement
    if (c_best < c_prev) {
      policy.CopyFrom(candidate_policy[0], horizon);
      // nominal
    } else {
      policy.CopyFrom(candidate_policy[num_trajectory - 1], horizon);
    }
  }

  // stop timer
  policy_update_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - policy_update_start)
          .count();

  // set timers
  nominal_compute_time = nominal_time;
  model_derivative_compute_time = model_derivative_time;
  cost_derivative_compute_time = cost_derivative_time;
  rollouts_compute_time = rollouts_time;
  backward_pass_compute_time = backward_pass_time;
  policy_update_compute_time = policy_update_time;
}

// compute trajectory using nominal policy
void iLQGPlanner::NominalTrajectory(int horizon) {
  // policy
  auto nominal_policy = [&cp = candidate_policy[0]](
                            double* action, const double* state, double time) {
    cp.Action(action, state, time);
  };

  // policy rollout
  trajectory[0].Rollout(nominal_policy, task, model, data_[0].get(),
                        state.data(), time, mocap.data(), horizon);
}

// set action from policy
void iLQGPlanner::ActionFromPolicy(double* action, const double* state,
                                   double time) {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  policy.Action(action, state, time);
}

// return trajectory with best total return
const Trajectory* iLQGPlanner::BestTrajectory() {
  const std::shared_lock<std::shared_mutex> lock(mtx_);
  return &policy.trajectory;
}

// visualize planner-specific traces in GUI
void iLQGPlanner::Traces(mjvScene* scn) {}

// planner-specific GUI elements
void iLQGPlanner::GUI(mjUI& ui) {
  mjuiDef defiLQG[] = {
      {mjITEM_SLIDERINT, "Rollouts", 2, &num_trajectory, "0 1"},
      // {mjITEM_RADIO, "Action Lmt.", 2, &settings.action_limits, "Off\nOn"},
      // {mjITEM_SLIDERINT, "Iterations", 2, &settings.max_rollout, "1 25"},
      {mjITEM_SELECT, "Policy Interp.", 2, &policy.representation,
       "Zero\nLinear\nCubic"},
      {mjITEM_SELECT, "Reg. Type", 2, &settings.regularization_type,
       "Control\nFeedback\nValue\nNone"},
      {mjITEM_END}};

  // set number of trajectory slider limits
  mju::sprintf_arr(defiLQG[0].other, "%i %i", 1, kMaxTrajectory);

  // add iLQG planner
  mjui_add(&ui, defiLQG);
}

// planner-specific plots
void iLQGPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                        int planning) {
  // bounds
  double planner_bounds[2] = {-6, 6};

  // ----- planner ----- //

  // regularization
  mjpc::PlotUpdateData(fig_planner, planner_bounds,
                       fig_planner->linedata[0][0] + 1,
                       mju_log10(mju_max(backward_pass.regularization, 1.0e-6)),
                       100, 0, 0, 1, -100);

  // step size
  mjpc::PlotUpdateData(
      fig_planner, planner_bounds, fig_planner->linedata[1][0] + 1,
      mju_log10(mju_max(step_size, 1.0e-6)), 100, 1, 0, 1, -100);

  // improvement
  // mjpc::PlotUpdateData(
  //     fig_planner, planner_bounds, fig_planner->linedata[2][0] + 1,
  //     mju_log10(mju_max(improvement, 1.0e-6)), 100, 2, 0, 1, -100);

  // // expected
  // mjpc::PlotUpdateData(
  //     fig_planner, planner_bounds, fig_planner->linedata[3][0] + 1,
  //     mju_log10(mju_max(expected, 1.0e-6)), 100, 3, 0, 1, -100);

  // // surprise
  // mjpc::PlotUpdateData(
  //     fig_planner, planner_bounds, fig_planner->linedata[4][0] + 1,
  //     mju_log10(mju_max(surprise, 1.0e-6)), 100, 4, 0, 1, -100);

  // legend
  mju::strcpy_arr(fig_planner->linename[0], "Regularization");
  mju::strcpy_arr(fig_planner->linename[1], "Step Size");
  // mju::strcpy_arr(fig_planner->linename[2], "Improvement");
  // mju::strcpy_arr(fig_planner->linename[3], "Expected");
  // mju::strcpy_arr(fig_planner->linename[4], "Surprise");

  // ranges
  fig_planner->range[1][0] = planner_bounds[0];
  fig_planner->range[1][1] = planner_bounds[1];

  // ----- timer ----- //
  double timer_bounds[2] = {0, 1};

  // update plots
  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[9][0] + 1,
                 1.0e-3 * nominal_compute_time * planning, 100, 9, 0, 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[10][0] + 1,
                 1.0e-3 * model_derivative_compute_time * planning, 100, 10, 0,
                 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[11][0] + 1,
                 1.0e-3 * cost_derivative_compute_time * planning, 100, 11, 0,
                 1, -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[12][0] + 1,
                 1.0e-3 * backward_pass_compute_time * planning, 100, 12, 0, 1,
                 -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[13][0] + 1,
                 1.0e-3 * rollouts_compute_time * planning, 100, 13, 0, 1,
                 -100);

  PlotUpdateData(fig_timer, timer_bounds, fig_timer->linedata[14][0] + 1,
                 1.0e-3 * policy_update_compute_time * planning, 100, 14, 0, 1,
                 -100);

  // legend
  mju::strcpy_arr(fig_timer->linename[9], "Nominal");
  mju::strcpy_arr(fig_timer->linename[10], "Model Deriv.");
  mju::strcpy_arr(fig_timer->linename[11], "Cost Deriv.");
  mju::strcpy_arr(fig_timer->linename[12], "Backward Pass");
  mju::strcpy_arr(fig_timer->linename[13], "Rollouts");
  mju::strcpy_arr(fig_timer->linename[14], "Policy Update");

  fig_timer->range[0][0] = -100;
  fig_timer->range[0][1] = 0;
  fig_timer->range[1][0] = 0.0;
  fig_timer->range[1][1] = timer_bounds[1];
}

// compute candidate trajectories
void iLQGPlanner::Rollouts(int horizon, ThreadPool& pool) {
  int count_before = pool.GetCount();
  for (int i = 0; i < num_trajectory; i++) {
    pool.Schedule([&data = data_, &trajectory = trajectory,
                   &candidate_policy = candidate_policy,
                   &improvement_step = improvement_step, &model = this->model,
                   &task = this->task, &state = this->state, &time = this->time,
                   &mocap = this->mocap, horizon, i]() {
      // scale improvement
      mju_addScl(candidate_policy[i].trajectory.actions.data(),
                 candidate_policy[i].trajectory.actions.data(),
                 candidate_policy[i].action_improvement.data(),
                 improvement_step[i], model->nu * horizon);

      // policy
      auto feedback_policy = [&candidate_policy = candidate_policy, model, i](
                                 double* action, const double* state,
                                 int index) {
        // dimensions
        int dim_state = model->nq + model->nv + model->na;
        int dim_state_derivative = 2 * model->nv + model->na;
        int dim_action = model->nu;

        // set improved action
        mju_copy(
            action,
            DataAt(candidate_policy[i].trajectory.actions, index * dim_action),
            dim_action);

        // ----- feedback ----- //

        // difference between current state and nominal state
        StateDiff(
            model, candidate_policy[i].state_scratch.data(),
            DataAt(candidate_policy[i].trajectory.states, index * dim_state),
            state, 1.0);

        // compute feedback term
        mju_mulMatVec(candidate_policy[i].action_scratch.data(),
                      DataAt(candidate_policy[i].feedback_gain,
                             index * dim_action * dim_state_derivative),
                      candidate_policy[i].state_scratch.data(), dim_action,
                      dim_state_derivative);

        // add feedback
        mju_addTo(action, candidate_policy[i].action_scratch.data(),
                  dim_action);

        // clamp controls
        Clamp(action, model->actuator_ctrlrange, dim_action);
      };

      // policy rollout (discrete time)
      trajectory[i].RolloutDiscrete(feedback_policy, task, model,
                                    data[ThreadPool::WorkerId()].get(),
                                    state.data(), time, mocap.data(), horizon);
    });
  }
  pool.WaitCount(count_before + num_trajectory);

  pool.ResetCount();
}

}  // namespace mjpc
