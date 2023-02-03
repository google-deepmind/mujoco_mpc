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

#include "planners/ilqs/planner.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>

#include "array_safety.h"
#include "planners/ilqg/planner.h"
#include "planners/linear_solve.h"
#include "planners/planner.h"
#include "planners/sampling/planner.h"
#include "states/state.h"
#include "trajectory.h"
#include "utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// initialize data and settings
void iLQSPlanner::Initialize(mjModel* model, const Task& task) {
  // Sampling
  sampling.Initialize(model, task);

  // iLQG
  ilqg.Initialize(model, task);
}

// allocate memory
void iLQSPlanner::Allocate() {
  // Sampling
  sampling.Allocate();

  // iLQG
  ilqg.Allocate();

  // ----- policy conversion ----- //
  // spline mapping
  for (auto& mapping : mappings) {
    mapping->Allocate(sampling.model->nu);
  }
}

// reset memory to zeros
void iLQSPlanner::Reset(int horizon) {
  // Sampling
  sampling.Reset(horizon);

  // iLQG
  ilqg.Reset(horizon);

  // online_policy
  online_policy = 0;
}

// set state
void iLQSPlanner::SetState(State& state) {
  // Sampling
  sampling.SetState(state);

  // iLQG
  ilqg.SetState(state);
}

// optimize nominal policy using iLQS
void iLQSPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // ----- Sampling ----- //
  // if num_trajectory_ has changed, use it in this new iteration.
  // num_trajectory_ might change while this function runs. Keep it constant
  // for the duration of this function.
  int num_trajectory = sampling.num_trajectory_;
  sampling.ResizeMjData(sampling.model, pool.NumThreads());

  // ----- nominal policy ----- //
  // start timer
  auto nominal_start = std::chrono::steady_clock::now();

  // copy nominal policy
  {
    const std::shared_lock<std::shared_mutex> lock(sampling.mtx_);
    sampling.policy.num_parameters =
        sampling.model->nu * sampling.policy.num_spline_points;  // set
    sampling.candidate_policy[0].CopyFrom(sampling.policy,
                                          sampling.policy.num_spline_points);
  }

  double* best_actions;
  double best_return;

  if (online_policy == 0) {
    // rollout old policy
    sampling.NominalTrajectory(horizon, pool);

    // set candidate policy nominal trajectory
    ilqg.candidate_policy[0].trajectory = sampling.trajectory[0];

    best_actions = sampling.trajectory[0].actions.data();
    best_return = sampling.trajectory[0].total_return;
  } else {
    // get nominal trajectory
    ilqg.NominalTrajectory(horizon, pool);

    // set nominal trajectory
    sampling.trajectory[0] = ilqg.trajectory[0];

    best_actions = ilqg.trajectory[0].actions.data();
    best_return = ilqg.trajectory[0].total_return;
  }

  // resample policy
  // TODO(taylorhowell): remove and only utilized new time trajectory
  sampling.ResamplePolicy(horizon);

  // get trajectory-parameter mapping
  // TODO(taylorhowell): compute only when necessary
  mappings[sampling.policy.representation]->Compute(
      sampling.candidate_policy[0].times.data(),
      sampling.candidate_policy[0].num_spline_points,
      sampling.trajectory[0].times.data(), sampling.trajectory[0].horizon - 1);

  // linear system solve
  if (solver.dim_row != sampling.model->nu * (horizon - 1) &&
      solver.dim_col !=
          sampling.model->nu * sampling.candidate_policy[0].num_spline_points) {
    solver.Initialize(
        sampling.model->nu * (horizon - 1),
        sampling.model->nu * sampling.candidate_policy[0].num_spline_points);
  }

  // TODO(taylorhowell): cheap version that reuses factorization if mapping hasn't changed
  solver.Solve(sampling.candidate_policy[0].parameters.data(),
               mappings[sampling.policy.representation]->Get(), best_actions);

  // clamp parameters
  for (int t = 0; t < sampling.candidate_policy[0].num_spline_points; t++) {
    Clamp(
        DataAt(sampling.candidate_policy[0].parameters, t * sampling.model->nu),
        sampling.model->actuator_ctrlrange, sampling.model->nu);
  }

  // stop timer
  sampling.nominal_compute_time = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - nominal_start)
                      .count();

  // ----- rollout noisy policies ----- //
  // start timer
  auto rollouts_start = std::chrono::steady_clock::now();

  // simulate noisy policies
  sampling.Rollouts(num_trajectory, horizon, pool);

  // ----- compare rollouts ----- //
  // reset
  sampling.winner = 0;

  // random search
  for (int i = 1; i < num_trajectory; i++) {
    if (sampling.trajectory[i].total_return <
        sampling.trajectory[sampling.winner].total_return) {
      sampling.winner = i;
    }
  }

  // stop timer
  sampling.rollouts_compute_time = std::chrono::duration_cast<std::chrono::microseconds>(
                       std::chrono::steady_clock::now() - rollouts_start)
                       .count();

  // ----- update policy ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();

  // copy best candidate policy
  {
    const std::shared_lock<std::shared_mutex> lock(sampling.mtx_);
    sampling.policy.CopyParametersFrom(
        sampling.candidate_policy[sampling.winner].parameters,
        sampling.candidate_policy[sampling.winner].times);
  }

  // stop timer
  sampling.policy_update_compute_time =
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - policy_update_start)
          .count();

  // improvement
  sampling.improvement = mju_max(
      best_return - sampling.trajectory[sampling.winner].total_return, 0.0);

  // check for improvement
  if (sampling.improvement > 0) {
    // set policy
    online_policy = 0;

    // set iLQG time to zero 
    ilqg.model_derivative_compute_time = 0.0;
    ilqg.cost_derivative_compute_time = 0.0;
    ilqg.rollouts_compute_time = 0.0;
    ilqg.backward_pass_compute_time = 0.0;
    ilqg.policy_update_compute_time = 0.0;

    return;
  }

  // ----- iLQG ----- //
  ilqg.Iteration(horizon, pool);

  // set policy
  if (ilqg.improvement > 0) {
    online_policy = 1;
  } else {
    online_policy = 0;
  }
}

// compute trajectory using nominal policy
void iLQSPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  if (online_policy == 0) {
    // Sampling
    sampling.NominalTrajectory(horizon, pool);
  } else {
    // iLQG
    ilqg.NominalTrajectory(horizon, pool);
  }
}

// set action from policy
void iLQSPlanner::ActionFromPolicy(double* action, const double* state,
                                   double time) {
  if (online_policy == 0) {
    // Sampling
    sampling.ActionFromPolicy(action, state, time);
  } else {
    // iLQG
    ilqg.ActionFromPolicy(action, state, time);
  }
}

// return trajectory with best total return
const Trajectory* iLQSPlanner::BestTrajectory() {
  if (online_policy == 0) {
    // Sampling
    return sampling.BestTrajectory();
  } else {
    // iLQG
    return ilqg.BestTrajectory();
  }
}

// visualize planner-specific traces in GUI
void iLQSPlanner::Traces(mjvScene* scn) {
  // visual sample traces from Sampling planner
  sampling.Traces(scn);
}

// planner-specific GUI elements
void iLQSPlanner::GUI(mjUI& ui) {
  // Sampling
  mju::sprintf_arr(ui.sect[5].item[10].name, "Sampling Settings");
  sampling.GUI(ui);

  // iLQG
  mjuiDef defiLQGSeparator[] = {{mjITEM_SEPARATOR, "iLQG Settings", 1},
                                {mjITEM_END}};
  mjui_add(&ui, defiLQGSeparator);
  ilqg.GUI(ui);
}

// planner-specific plots
void iLQSPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                        int planner_shift, int timer_shift, int planning) {
  // Sampling
  sampling.Plots(fig_planner, fig_timer, planner_shift, timer_shift, planning);

  // iLQG
  ilqg.Plots(fig_planner, fig_timer, planner_shift + 1, timer_shift + 4, planning);

  // ----- re-label ----- //
  // planner plots 
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improve. (S)");
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift + 1], "Reg. (LQ)");
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift + 2], "Step Size (LQ)");

  // timer plots 
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Nominal (S)");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Noise (S)");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Rollout (S)");
  mju::strcpy_arr(fig_timer->linename[3 + timer_shift], "Policy Update (S)");
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift + 4], "Nominal (LQ)");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift + 4], "Model Deriv. (LQ)");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift + 4], "Cost Deriv. (LQ)");
  mju::strcpy_arr(fig_timer->linename[3 + timer_shift + 4], "Backward Pass (LQ)");
  mju::strcpy_arr(fig_timer->linename[4 + timer_shift + 4], "Rollouts (LQ)");
  mju::strcpy_arr(fig_timer->linename[5 + timer_shift + 4], "Policy Update (LQ)");
}

}  // namespace mjpc
