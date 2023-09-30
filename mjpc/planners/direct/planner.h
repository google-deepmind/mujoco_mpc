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

#ifndef MJPC_PLANNERS_DIRECT_H_
#define MJPC_PLANNERS_DIRECT_H_

#include <shared_mutex>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/planners/direct/policy.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"

namespace mjpc {

// planner for iLQG
class DirectPlanner : public Planner {
 public:
  // constructor
  DirectPlanner() = default;

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon) override;

  // set state
  void SetState(const State& state) override;

  // optimize nominal policy using iLQG
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // set action from policy
  // if state == nullptr, return the nominal action without a feedback term
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces in GUI
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // ----- members ----- //
  mjModel* model;
  const Task* task;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_DIRECT_H_
