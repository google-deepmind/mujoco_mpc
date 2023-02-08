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

#ifndef MJPC_PLANNERS_OPTIMIZER_H_
#define MJPC_PLANNERS_OPTIMIZER_H_

#include <mujoco/mujoco.h>

#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

inline constexpr int kMaxTrajectory = 128;

// virtual planner
class Planner {
 public:
  // destructor
  virtual ~Planner() = default;

  // initialize data and settings
  virtual void Initialize(mjModel* model, const Task& task) = 0;

  // allocate memory
  virtual void Allocate() = 0;

  // reset memory to zeros
  virtual void Reset(int horizon) = 0;

  // set state
  virtual void SetState(State& state) = 0;

  // optimize nominal policy
  virtual void OptimizePolicy(int horizon, ThreadPool& pool) = 0;

  // compute trajectory using nominal policy
  virtual void NominalTrajectory(int horizon, ThreadPool& pool) = 0;

  // set action from policy
  virtual void ActionFromPolicy(double* action, const double* state,
                                double time) = 0;

  // return trajectory with best total return
  virtual const Trajectory* BestTrajectory() = 0;

  // visualize planner-specific traces
  virtual void Traces(mjvScene* scn) = 0;

  // planner-specific GUI elements
  virtual void GUI(mjUI& ui) = 0;

  // planner-specific plots
  virtual void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                     int planner_shift, int timer_shift, int planning) = 0;

  std::vector<UniqueMjData> data_;
  void ResizeMjData(const mjModel* model, int num_threads);
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_OPTIMIZER_H_
