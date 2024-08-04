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

#include "mjpc/planners/ilqs/planner.h"

#include <chrono>
#include <vector>

#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

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

  // mapping dimensions
  dim_actions = 0;
  dim_parameters = 0;
}

// reset memory to zeros
void iLQSPlanner::Reset(int horizon, const double* initial_repeated_action) {
  // Sampling
  sampling.Reset(horizon, initial_repeated_action);

  // iLQG
  ilqg.Reset(horizon, initial_repeated_action);

  // active_policy
  active_policy = kSampling;
  previous_active_policy = kSampling;

  // mapping dimensions
  dim_actions = 0;
  dim_parameters = 0;
}

// set state
void iLQSPlanner::SetState(const State& state) {
  // Sampling
  sampling.SetState(state);

  // iLQG
  ilqg.SetState(state);
}

// optimize nominal policy using iLQS
void iLQSPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  previous_active_policy = active_policy;
  ilqg.UpdateNumTrajectoriesFromGUI();
  if (previous_active_policy == kiLQG) {
    // In order to optimize via sampling, we first convert the traj-based policy
    // representation of iLQG (the previous winner) to a spline representation.

    ilqg.NominalTrajectory(horizon, pool);

    // ----- spline parameters from trajectory ----- //
    // get number of spline points
    int num_spline_points = sampling.policy.num_spline_points;

    // get times for spline parameters
    double nominal_time = sampling.time;
    double time_shift = mju_max(
        (horizon - 1) * sampling.model->opt.timestep / (num_spline_points - 1),
        1.0e-5);

    // get spline points
    spline_times_cache.clear();
    for (int t = 0; t < num_spline_points; t++) {
      spline_times_cache.push_back(nominal_time);
      nominal_time += time_shift;
    }

    // linear system solve
    if (dim_actions != sampling.model->nu * (horizon - 1) ||
        dim_parameters != sampling.model->nu * num_spline_points) {
      // dimension
      dim_parameters = sampling.model->nu * num_spline_points;
      dim_actions = sampling.model->nu * (horizon - 1);

      // compute parameter to action mapping
      mappings[sampling.policy.plan.Interpolation()]->Compute(
          spline_times_cache, num_spline_points,
          ilqg.candidate_policy[0].trajectory.times.data(), horizon - 1);

      // ----- compute inverse mapping ----- //
      // resize
      inversemapping_cache.resize(dim_parameters * dim_parameters);
      inversemapping.resize(dim_parameters * dim_actions);
      inversemappingT.resize(dim_actions * dim_parameters);

      // M = A' A
      double* mapping = mappings[sampling.policy.plan.Interpolation()]->Get();
      mju_mulMatTMat(inversemapping_cache.data(), mapping, mapping, dim_actions,
                     dim_parameters, dim_parameters);

      // cholesky(M)
      mju_cholFactor(inversemapping_cache.data(), dim_parameters, 0.0);

      // M \ A'
      for (int i = 0; i < dim_actions; i++) {
        mju_cholSolve(inversemappingT.data() + i * dim_parameters,
                      inversemapping_cache.data(), mapping + i * dim_parameters,
                      dim_parameters);
      }

      // transpose
      mju_transpose(inversemapping.data(), inversemappingT.data(), dim_actions,
                    dim_parameters);
    }

    // compute parameters from actions via inverse mapping
    spline_parameters_cache.resize(dim_parameters);
    mju_mulMatVec(spline_parameters_cache.data(), inversemapping.data(),
                  ilqg.candidate_policy[0].trajectory.actions.data(),
                  dim_parameters, dim_actions);

    // clamp parameters
    for (int t = 0; t < num_spline_points; t++) {
      Clamp(DataAt(spline_parameters_cache, t * sampling.model->nu),
            sampling.model->actuator_ctrlrange, sampling.model->nu);
    }
    sampling.policy.plan.Clear();
    for (int t = 0; t < num_spline_points; t++) {
      sampling.policy.plan.AddNode(
          spline_times_cache[t],
          absl::MakeConstSpan(
              spline_parameters_cache.data() + t * sampling.model->nu,
              sampling.model->nu));
    }
  }

  // try sampling
  sampling.OptimizePolicy(horizon, pool);

  // check for improvement
  if (sampling.winner > 0 &&  // if winner==0, there was surely no improvement
      (sampling.trajectory[sampling.winner].total_return <
       (previous_active_policy == kSampling
            ? sampling.trajectory[0].total_return
            : ilqg.candidate_policy[0].trajectory.total_return))) {
    // zero ilqg timers
    if (active_policy == kSampling) {
      ilqg.nominal_compute_time = 0.0;
    }
    ilqg.model_derivative_compute_time = 0.0;
    ilqg.cost_derivative_compute_time = 0.0;
    ilqg.backward_pass_compute_time = 0.0;
    ilqg.rollouts_compute_time = 0.0;
    ilqg.policy_update_compute_time = 0.0;

    active_policy = kSampling;

    // best rollout is from sampling, terminate early
    return;
  } else {  // no improvement found with sampling this round
    if (previous_active_policy == kSampling) {
      // update iLQG with the last winner
      ilqg.candidate_policy[0].trajectory = sampling.trajectory[0];
    }
  }

  // iLQG
  ilqg.Iteration(horizon, pool);

  // comparison for new active policy
  if (ilqg.trajectory[ilqg.winner].total_return <
      (previous_active_policy == kSampling
           ? sampling.trajectory[sampling.winner].total_return
           : ilqg.trajectory[0].total_return)) {
    active_policy = kiLQG;
  }
  // If no improvement was found either way, both policies were updated, but
  // active_policy is not.
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
                                   double time, bool use_previous) {
  if (use_previous) {
    if (previous_active_policy == kSampling) {
      // We always call sampling.OptimizePolicy above, which always updates the
      // sampling policy, so we always want the previous policy.
      sampling.ActionFromPolicy(action, state, time, true);
    } else {  // previous active policy was iLQG
      if (active_policy == kSampling) {
        // The most recent planner terminated early, so iLQG was not updated,
        // and the previous policy is iLQG's current policy.
        ilqg.ActionFromPolicy(action, state, time, false);
      } else {
        // Most recent planner step updated iLQG, so the previous policy is
        // iLQG's previous policy.
        ilqg.ActionFromPolicy(action, state, time, true);
      }
    }
  } else {  // use current policy
    if (active_policy == kSampling) {
      sampling.ActionFromPolicy(action, state, time, false);
    } else {
      ilqg.ActionFromPolicy(action, state, time, false);
    }
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
                        int planner_shift, int timer_shift, int planning,
                        int* shift) {
  // Sampling
  sampling.Plots(fig_planner, fig_timer, planner_shift, timer_shift, planning,
                 shift);

  // iLQG
  ilqg.Plots(fig_planner, fig_timer, planner_shift + 1, timer_shift + 4,
             planning, shift);

  // ----- re-label ----- //
  // planner plots
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift], "Improve. (S)");
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift + 1], "Reg. (LQ)");
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift + 2],
                  "Action Step (LQ)");
  mju::strcpy_arr(fig_planner->linename[0 + planner_shift + 3],
                  "Feedback Scaling (LQ)");

  // timer plots
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift], "Noise (S)");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift], "Rollout (S)");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift], "Policy Update (S)");
  mju::strcpy_arr(fig_timer->linename[0 + timer_shift + 3], "Nominal (LQ)");
  mju::strcpy_arr(fig_timer->linename[1 + timer_shift + 3],
                  "Model Deriv. (LQ)");
  mju::strcpy_arr(fig_timer->linename[2 + timer_shift + 3], "Cost Deriv. (LQ)");
  mju::strcpy_arr(fig_timer->linename[3 + timer_shift + 3],
                  "Backward Pass (LQ)");
  mju::strcpy_arr(fig_timer->linename[4 + timer_shift + 3], "Rollouts (LQ)");
  mju::strcpy_arr(fig_timer->linename[5 + timer_shift + 3],
                  "Policy Update (LQ)");
}

}  // namespace mjpc
