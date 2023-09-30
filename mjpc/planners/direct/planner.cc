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

#include "mjpc/planners/direct/planner.h"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <mutex>

#include "mjpc/array_safety.h"
#include "mjpc/planners/direct/policy.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

// initialize data and settings
void DirectPlanner::Initialize(mjModel* model, const Task& task) {
  // model
  this->model = model;

  // task
  this->task = &task;
}

// allocate memory
void DirectPlanner::Allocate() {}

// reset memory to zeros
void DirectPlanner::Reset(int horizon) {}

// set state
void DirectPlanner::SetState(const State& state) {}

// optimize nominal policy using direct
void DirectPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {}

// set action from policy
void DirectPlanner::ActionFromPolicy(double* action, const double* state,
                                     double time, bool use_previous) {}

// return trajectory with best total return
const Trajectory* DirectPlanner::BestTrajectory() { return NULL; }

// visualize planner-specific traces in GUI
void DirectPlanner::Traces(mjvScene* scn) {}

// planner-specific GUI elements
void DirectPlanner::GUI(mjUI& ui) {
  mjuiDef defDirect[] = {{mjITEM_END}};

  // add Direct planner
  mjui_add(&ui, defDirect);
}

// planner-specific plots
void DirectPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                          int planner_shift, int timer_shift, int planning,
                          int* shift) {}

}  // namespace mjpc
