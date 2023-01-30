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
  parameter_matrix_scratch.resize((kMaxTrajectoryHorizon * sampling.model->nu) *
                                  (kMaxTrajectoryHorizon * sampling.model->nu));
  parameter_vector_scratch.resize(kMaxTrajectoryHorizon * sampling.model->nu);
}

// reset memory to zeros
void iLQSPlanner::Reset(int horizon) {
  // Sampling
  sampling.Reset(horizon);

  // iLQG
  ilqg.Reset(horizon);

  // winner
  winner = 0;

  // ----- policy conversion ----- //
  std::fill(parameter_matrix_scratch.begin(),
            parameter_matrix_scratch.begin() +
                (kMaxTrajectoryHorizon * sampling.model->nu) *
                    (kMaxTrajectoryHorizon * sampling.model->nu),
            0.0);
  std::fill(parameter_vector_scratch.begin(),
            parameter_vector_scratch.begin() +
                (kMaxTrajectoryHorizon * sampling.model->nu),
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
  // Sampling
  sampling.OptimizePolicy(horizon, pool);

  // iLQG
  // ilqg.OptimizePolicy(horizon, pool);
}

// compute trajectory using nominal policy
void iLQSPlanner::NominalTrajectory(int horizon) {
  if (winner == 0) {
    // Sampling
    sampling.NominalTrajectory(horizon);
  } else {
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
  // sampling.Traces(scn);

  // iLQG
  // ilqg.Traces(scn);
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
                        int planning) {
  // Sampling
  // sampling.Plots(fig_planner, fig_timer, planning);

  // iLQG
  // ilqg.Plots(fig_planner, fig_timer, planning);
}

}  // namespace mjpc
