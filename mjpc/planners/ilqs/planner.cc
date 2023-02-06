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

// iLQS planners
enum iLQSPlanners : int {
  kSampling = 0,
  kiLQG,
};

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

  // active_policy
  active_policy = kSampling;
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
  if (active_policy == kiLQG) {
    // get nominal trajectory
    ilqg.NominalTrajectory(horizon, pool);

    // ----- spline parameters from trajectory ----- //
    // get number of spline points
    int num_spline_points = sampling.policy.num_spline_points;

    // get times for spline parameters
    double nominal_time = sampling.time;
    double time_shift = mju_max(
        (horizon - 1) * sampling.model->opt.timestep / (num_spline_points - 1), 1.0e-5);

    // get spline points
    for (int t = 0; t < num_spline_points; t++) {
      sampling.policy.times[t] = nominal_time;
      nominal_time += time_shift;
    }

    // time power transformation
    PowerSequence(sampling.policy.times.data(), time_shift,
                  sampling.policy.times[0],
                  sampling.policy.times[num_spline_points - 1],
                  sampling.timestep_power, num_spline_points);

    // linear system solve
    if (solver.dim_row != sampling.model->nu * (horizon - 1) &&
        solver.dim_col !=
            sampling.model->nu * num_spline_points) {
      mappings[sampling.policy.representation]->Compute(
          sampling.policy.times.data(),
          num_spline_points,
          ilqg.candidate_policy[0].trajectory.times.data(), horizon - 1);
      solver.Initialize(
          sampling.model->nu * (horizon - 1),
          sampling.model->nu * num_spline_points);
    }

    // TODO(taylorhowell): cheap version that reuses factorization if mapping hasn't changed
    solver.Solve(sampling.policy.parameters.data(),
                mappings[sampling.policy.representation]->Get(), ilqg.candidate_policy[0].trajectory.actions.data());

    // clamp parameters
    for (int t = 0; t < num_spline_points; t++) {
      Clamp(
          DataAt(sampling.policy.parameters, t * sampling.model->nu),
          sampling.model->actuator_ctrlrange, sampling.model->nu);
    }
  }

  // update policy
  sampling.OptimizePolicy(horizon, pool);

  // check for improvement
  if (active_policy == kSampling) {
    if (sampling.improvement > 0) {
      return; 
    } else {
      // set iLQG nominal trajectory
      ilqg.candidate_policy[0].trajectory = sampling.trajectory[0];
    }
  } 
  
  // iLQG
  ilqg.Iteration(horizon, pool);
  if (ilqg.trajectory[ilqg.winner].total_return < sampling.trajectory[sampling.winner].total_return) {
    active_policy = kiLQG;
  } else {
    active_policy = kSampling;
  }
}

// compute trajectory using nominal policy
void iLQSPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  if (active_policy == kSampling) {
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
  if (active_policy == kSampling) {
    // Sampling
    sampling.ActionFromPolicy(action, state, time);
  } else {
    // iLQG
    ilqg.ActionFromPolicy(action, state, time);
  }
}

// return trajectory with best total return
const Trajectory* iLQSPlanner::BestTrajectory() {
  // return &trajectory;
  if (active_policy == kSampling) {
    return sampling.BestTrajectory();
  } else {
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
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift + 2], "Action Step (LQ)");
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift + 3], "Feedback Scaling (LQ)");

  // timer plots 
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise (S)");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout (S)");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update (S)");
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift + 3], "Nominal (LQ)");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift + 3], "Model Deriv. (LQ)");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift + 3], "Cost Deriv. (LQ)");
  mju::strcpy_arr(fig_timer->linename[3 + timer_shift + 3], "Backward Pass (LQ)");
  mju::strcpy_arr(fig_timer->linename[4 + timer_shift + 3], "Rollouts (LQ)");
  mju::strcpy_arr(fig_timer->linename[5 + timer_shift + 3], "Policy Update (LQ)");
}

}  // namespace mjpc
