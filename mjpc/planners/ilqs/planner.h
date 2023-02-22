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

#ifndef MJPC_PLANNERS_ILQS_OPTIMIZER_H_
#define MJPC_PLANNERS_ILQS_OPTIMIZER_H_

#include <mujoco/mujoco.h>

#include <shared_mutex>
#include <vector>

#include "mjpc/planners/gradient/spline_mapping.h"
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/linear_solve.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// planner for iLQS
class iLQSPlanner : public Planner {
 public:
  // constructor
  iLQSPlanner() {
    // spline mapping linear operators for policy conversion
    mappings.emplace_back(new ZeroSplineMapping);
    mappings.emplace_back(new LinearSplineMapping);
    mappings.emplace_back(new CubicSplineMapping);
  }

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon) override;

  // set state
  void SetState(State& state) override;

  // optimize nominal policy using iLQS
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // set action from policy
  void ActionFromPolicy(double* action, const double* state,
                        double time) override;

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces in GUI
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning) override;

  // ----- planners ----- //
  SamplingPlanner sampling;
  iLQGPlanner ilqg;

  // ----- policy conversion ----- //
  // spline mapping
  std::vector<std::unique_ptr<SplineMapping>> mappings;

  // inverse mapping
  std::vector<double> inversemapping_cache;
  std::vector<double> inversemappingT;
  std::vector<double> inversemapping;

  // mapping dimensions
  int dim_actions;
  int dim_parameters;

  // online policy for returning actions
  int active_policy;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ILQS_OPTIMIZER_H_
