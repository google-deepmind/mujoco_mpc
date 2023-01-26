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
  // scratch
  parameter_matrix_scratch.resize((kMaxTrajectoryHorizon * sampling.model->nu) * (kMaxTrajectoryHorizon * sampling.model->nu));
  parameter_vector_scratch.resize(kMaxTrajectoryHorizon * sampling.model->nu);
}

// reset memory to zeros
void iLQSPlanner::Reset(int horizon) {
  // Sampling 
  sampling.Reset(horizon);

  // iLQG
  ilqg.Reset(horizon);

  // winner
  winner = 1;

  // ----- policy conversion ----- //
  std::fill(parameter_matrix_scratch.begin(), 
            parameter_matrix_scratch.begin() + (kMaxTrajectoryHorizon * sampling.model->nu) * (kMaxTrajectoryHorizon * sampling.model->nu), 
            0.0);
  std::fill(parameter_vector_scratch.begin(), 
            parameter_vector_scratch.begin() + (kMaxTrajectoryHorizon * sampling.model->nu), 
            0.0);
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
  // iLQS
  // TODO(taylorhowell): allocate less data, half?
  ilqg.ResizeMjData(ilqg.model, pool.NumThreads());
  sampling.ResizeMjData(sampling.model, pool.NumThreads());

  // timers
  double nominal_time = 0.0;
  double model_derivative_time = 0.0;
  double cost_derivative_time = 0.0;
  double rollouts_time = 0.0;
  double backward_pass_time = 0.0;
  double policy_update_time = 0.0;

  // maximum number of trajectories in linesearch
  ilqg.num_trajectory = mju_min(ilqg.num_trajectory, kMaxTrajectory);

  // ----- nominal rollout ----- //
  // start timer
  auto nominal_start = std::chrono::steady_clock::now();

  // previous best cost
  double c_prev;

  // Sampling is current policy
  if (winner == 0) {
    // ----- new policy parameters for Sampling ----- //
    // copy nominal policy
    {
      const std::shared_lock<std::shared_mutex> lock(sampling.mtx_);
      sampling.policy.num_parameters = sampling.model->nu * sampling.policy.num_spline_points;  // set
      sampling.candidate_policy[0].CopyFrom(sampling.policy, sampling.policy.num_spline_points);
    }

    // rollout nominal policy
    sampling.NominalTrajectory(horizon);
    if (sampling.trajectory[0].failure) {
      std::cerr << "Nominal trajectory diverged.\n";
    }

    // approximate new sampling policy
    sampling.ResamplePolicy(horizon);

    // set previous best cost
    c_prev = sampling.trajectory[sampling.winner].total_return;

    // set ilqg nominal trajectory
    ilqg.candidate_policy[0].trajectory = sampling.trajectory[0];

  // iLQG is current policy
  } else {
    // ----- new nominal trajectory for iLQG ----- //
    // copy nominal policy
    {
      const std::shared_lock<std::shared_mutex> lock(ilqg.mtx_);
      ilqg.candidate_policy[0].CopyFrom(ilqg.policy, horizon);
      ilqg.candidate_policy[0].representation = ilqg.policy.representation;
    }

    // rollout nominal policy
    ilqg.NominalTrajectory(horizon);
    if (ilqg.trajectory[0].failure) {
      std::cerr << "Nominal trajectory diverged.\n";
    }

    // set previous best cost
    c_prev = ilqg.trajectory[0].total_return;

    // set candidate policy nominal trajectory
    ilqg.candidate_policy[0].trajectory = ilqg.trajectory[0];

    // ----- policy conversion - trajectory -> parameters ----- //
    // copy nominal policy
    {
      const std::shared_lock<std::shared_mutex> lock(sampling.mtx_);
      sampling.policy.num_parameters = sampling.model->nu * sampling.policy.num_spline_points;  // set
      sampling.candidate_policy[0].CopyFrom(sampling.policy, sampling.policy.num_spline_points);
    }

    // compute spline mapping linear operator
    mappings[sampling.policy.representation]->Compute(
        sampling.candidate_policy[0].times.data(), sampling.candidate_policy[0].num_spline_points, 
        sampling.trajectory[0].times.data(), sampling.trajectory[0].horizon - 1);

    // recover parameters from trajectory via least-squares
    //   A = Mapping' Mapping
    mju_mulMatTMat(parameter_matrix_scratch.data(), 
                  mappings[sampling.policy.representation]->Get(), 
                  mappings[sampling.policy.representation]->Get(), 
                  sampling.model->nu * (horizon - 1),
                  sampling.model->nu * sampling.candidate_policy[0].num_spline_points,
                  sampling.model->nu * sampling.candidate_policy[0].num_spline_points);

    // factorization 
    mju_cholFactor(parameter_matrix_scratch.data(), 
                  sampling.model->nu * sampling.candidate_policy[0].num_spline_points, 
                  0.0);

    // parameter vector scratch 
    //   Parameters = Mapping' Trajectory
    mju_mulMatTVec(parameter_vector_scratch.data(), 
                  mappings[sampling.policy.representation]->Get(), 
                  ilqg.candidate_policy[0].trajectory.actions.data(), 
                  sampling.model->nu * (horizon - 1),
                  sampling.model->nu * sampling.candidate_policy[0].num_spline_points);

    // compute parameters 
    mju_cholSolve(sampling.candidate_policy[0].parameters.data(), 
                  parameter_matrix_scratch.data(), parameter_vector_scratch.data(), 
                  sampling.model->nu * sampling.candidate_policy[0].num_spline_points);

    // clamp parameters
    for (int t = 0; t < sampling.candidate_policy[0].num_spline_points; t++) {
      Clamp(DataAt(sampling.candidate_policy[0].parameters, t * sampling.model->nu),
            sampling.model->actuator_ctrlrange, sampling.model->nu);
    }

    printf("mapping: \n");
    mju_printMat(mappings[sampling.policy.representation]->Get(), sampling.model->nu * (horizon - 1),
                  sampling.model->nu * sampling.policy.num_spline_points);

    printf("recovered parameters: \n");
    mju_printMat(sampling.candidate_policy[0].parameters.data(), 
                 sampling.candidate_policy[0].num_spline_points, 
                 sampling.model->nu);
  }

  // end timer
  nominal_time = std::chrono::duration_cast<std::chrono::microseconds>(
                     std::chrono::steady_clock::now() - nominal_start)
                     .count();

  // ----- iLQG ----- //

  // ----- model derivatives ----- //
  // start timer
  auto model_derivative_start = std::chrono::steady_clock::now();

  // compute model and sensor Jacobians
  ilqg.model_derivative.Compute(
      ilqg.model, ilqg.data_, ilqg.candidate_policy[0].trajectory.states.data(),
      ilqg.candidate_policy[0].trajectory.actions.data(),
      ilqg.candidate_policy[0].trajectory.times.data(), ilqg.dim_state,
      ilqg.dim_state_derivative, ilqg.dim_action, ilqg.dim_sensor, horizon,
      ilqg.settings.fd_tolerance, ilqg.settings.fd_mode, pool);

  // stop timer
  model_derivative_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - model_derivative_start)
          .count();

  // ----- cost derivatives ----- //
  // start timer
  auto cost_derivative_start = std::chrono::steady_clock::now();

  // cost derivatives
  ilqg.cost_derivative.Compute(
      ilqg.candidate_policy[0].trajectory.residual.data(),
      ilqg.model_derivative.C.data(), ilqg.model_derivative.D.data(),
      ilqg.dim_state_derivative, ilqg.dim_action, ilqg.dim_max, ilqg.dim_sensor,
      ilqg.task->num_residual, ilqg.task->dim_norm_residual.data(), ilqg.task->num_cost,
      ilqg.task->weight.data(), ilqg.task->norm.data(), ilqg.task->num_parameter.data(),
      ilqg.task->num_norm_parameter.data(), ilqg.task->risk, horizon, pool);

  // end timer
  cost_derivative_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - cost_derivative_start)
          .count();

  // ----- backward pass ----- //
  // start timer
  auto backward_pass_start = std::chrono::steady_clock::now();

  // compute feedback gains and action improvement via Riccati
  ilqg.backward_pass.Riccati(&ilqg.candidate_policy[0], &ilqg.model_derivative,
                        &ilqg.cost_derivative, ilqg.dim_state_derivative, ilqg.dim_action,
                        horizon, ilqg.backward_pass.regularization, ilqg.boxqp,
                        ilqg.candidate_policy[0].trajectory.actions.data(),
                        ilqg.model->actuator_ctrlrange, ilqg.settings);

  // end timer
  backward_pass_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - backward_pass_start)
          .count();

  // ----- rollout policy ----- //
  auto rollouts_start = std::chrono::steady_clock::now();

  // copy policy
  for (int j = 1; j < ilqg.num_trajectory; j++) {
    ilqg.candidate_policy[j].CopyFrom(ilqg.candidate_policy[0], horizon);
    ilqg.candidate_policy[j].representation = ilqg.candidate_policy[0].representation;
  }

  // improvement step sizes (log scaling)
  LogScale(ilqg.improvement_step, 1.0, ilqg.settings.min_step_size, ilqg.num_trajectory - 1);
  ilqg.improvement_step[ilqg.num_trajectory - 1] = 0.0;

  // feedback rollouts (parallel)
  ilqg.Rollouts(horizon, pool);

  // ----- Sampling ----- //
  // simulate noisy policies
  sampling.Rollouts(sampling.num_trajectory_, horizon, pool);

  // ----- evaluate rollouts ------ //
  double c_best = c_prev;
  ilqg.winner = ilqg.num_trajectory - 1;
  int failed = 0;
  if (ilqg.trajectory[ilqg.num_trajectory - 1].failure) {
    failed++;
  }
  for (int j = ilqg.num_trajectory - 2; j >= 0; j--) {
    if (ilqg.trajectory[j].failure) {
      failed++;
      continue;
    }
    // compute cost
    double c_sample = ilqg.trajectory[j].total_return;

    // compare cost
    if (c_sample < c_best) {
      c_best = c_sample;
      ilqg.winner = j;
    }
  }
  if (failed) {
    std::cerr << "iLQG: " << failed << " out of " << ilqg.num_trajectory
              << " rollouts failed.\n";
  }

  // update nominal with winner
  ilqg.candidate_policy[0].trajectory = ilqg.trajectory[ilqg.winner];

  // improvement
  ilqg.step_size = ilqg.improvement_step[ilqg.winner];
  ilqg.expected = -1.0 * ilqg.step_size *
                  (ilqg.backward_pass.dV[0] + ilqg.step_size * ilqg.backward_pass.dV[1]) +
              1.0e-16;
  ilqg.improvement = c_prev - c_best;
  ilqg.surprise = mju_min(mju_max(0, ilqg.improvement / ilqg.expected), 2);

  // update regularization
  ilqg.backward_pass.UpdateRegularization(ilqg.settings.min_regularization,
                                      ilqg.settings.max_regularization, ilqg.surprise,
                                      ilqg.step_size);

  // stop timer
  rollouts_time += std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - rollouts_start)
                        .count();
  
  // ----- policy update ----- //
  // start timer
  auto policy_update_start = std::chrono::steady_clock::now();
  {
    const std::shared_lock<std::shared_mutex> lock(ilqg.mtx_);
    // improvement
    if (c_best < c_prev) {
      ilqg.policy.CopyFrom(ilqg.candidate_policy[0], horizon);
      // nominal
    } else {
      ilqg.policy.CopyFrom(ilqg.candidate_policy[ilqg.num_trajectory - 1], horizon);
    }
  }

  // stop timer
  policy_update_time +=
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::steady_clock::now() - policy_update_start)
          .count();

  // set timers
  ilqg.nominal_compute_time = nominal_time;
  ilqg.model_derivative_compute_time = model_derivative_time;
  ilqg.cost_derivative_compute_time = cost_derivative_time;
  ilqg.rollouts_compute_time = rollouts_time;
  ilqg.backward_pass_compute_time = backward_pass_time;
  ilqg.policy_update_compute_time = policy_update_time;

  // ----- Sampling ----- //
  // // start timer
  // auto rollouts_start = std::chrono::steady_clock::now();

  // // simulate noisy policies
  // sampling.Rollouts(sampling.num_trajectory_, horizon, pool);

  // // ----- compare rollouts ----- //
  // // reset
  // sampling.winner = 0;

  // // random search
  // for (int i = 1; i < sampling.num_trajectory_; i++) {
  //   if (sampling.trajectory[i].total_return < sampling.trajectory[sampling.winner].total_return) {
  //     sampling.winner = i;
  //   }
  // }

  // // stop timer
  // rollouts_time += std::chrono::duration_cast<std::chrono::microseconds>(
  //                      std::chrono::steady_clock::now() - rollouts_start)
  //                      .count();

  // // ----- update policy ----- //
  // // start timer
  // auto policy_update_start = std::chrono::steady_clock::now();

  // // copy best candidate policy
  // {
  //   const std::shared_lock<std::shared_mutex> lock(sampling.mtx_);
  //   sampling.policy.CopyParametersFrom(sampling.candidate_policy[sampling.winner].parameters,
  //                   sampling.candidate_policy[sampling.winner].times);
  // }

  // // stop timer
  // policy_update_time +=
  //     std::chrono::duration_cast<std::chrono::microseconds>(
  //         std::chrono::steady_clock::now() - policy_update_start)
  //         .count();

  // // improvement
  // sampling.improvement = mju_max(c_prev - sampling.trajectory[sampling.winner].total_return, 0.0);

  // // set timers
  // sampling.nominal_compute_time = nominal_time;
  // sampling.rollouts_compute_time = rollouts_time;
  // sampling.policy_update_compute_time = policy_update_time;
}

// compute trajectory using nominal policy
void iLQSPlanner::NominalTrajectory(int horizon) {
  if (winner == 0) {
    // Sampling 
    sampling.NominalTrajectory(horizon);
  } else  {
    // iLQG 
    ilqg.NominalTrajectory(horizon);
  }
}

// set action from policy
void iLQSPlanner::ActionFromPolicy(double* action, const double* state,
                                   double time) {
  if (winner == 0) {
    // Sampling 
    sampling.ActionFromPolicy(action, state, time);
  } else {
    // iLQG 
    ilqg.ActionFromPolicy(action, state, time);
  }
}

// return trajectory with best total return
const Trajectory* iLQSPlanner::BestTrajectory() {
  if (winner == 0) {
    // Sampling 
    return sampling.BestTrajectory();
  } else {
    // iLQG
    return ilqg.BestTrajectory();
  }
}

// visualize planner-specific traces in GUI
void iLQSPlanner::Traces(mjvScene* scn) {
  // Sampling 
  sampling.Traces(scn);

  // iLQG 
  // ilqg.Traces(scn);
}

// planner-specific GUI elements
void iLQSPlanner::GUI(mjUI& ui) {
  // Sampling 
  mju::sprintf_arr(ui.sect[5].item[10].name, "Sampling Settings");
  sampling.GUI(ui);

  // iLQG 
  mjuiDef defiLQGSeparator[] = {
      {mjITEM_SEPARATOR, "iLQG Settings", 1},
      {mjITEM_END}};
  mjui_add(&ui, defiLQGSeparator);
  ilqg.GUI(ui);
}

// planner-specific plots
void iLQSPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                        int planning) {
  // Sampling 
  sampling.Plots(fig_planner, fig_timer, planning);

  // iLQG
  // ilqg.Plots(fig_planner, fig_timer, planning);
}

}  // namespace mjpc
