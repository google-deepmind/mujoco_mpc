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

#ifndef MJPC_PLANNERS_ILQG_OPTIMIZER_H_
#define MJPC_PLANNERS_ILQG_OPTIMIZER_H_

#include <shared_mutex>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/planners/ilqg/backward_pass.h"
#include "mjpc/planners/ilqg/policy.h"
#include "mjpc/planners/ilqg/settings.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// planner for iLQG
class iLQGPlanner : public Planner {
 public:
  // constructor
  iLQGPlanner() = default;

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon) override;

  // set state
  void SetState(State& state) override;

  // optimize nominal policy using iLQG
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

  // single iLQG iteration
  void Iteration(int horizon, ThreadPool& pool);

  // linesearch over action improvement
  void ActionRollouts(int horizon, ThreadPool& pool);

  // linesearch over feedback scaling
  void FeedbackRollouts(int horizon, ThreadPool& pool);

  // return index of trajectory with best rollout
  int BestRollout(double previous_return, int num_trajectory);

  //

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  iLQGPolicy policy;
  iLQGPolicy candidate_policy[kMaxTrajectory];

  // dimensions
  int dim_state;             // state
  int dim_state_derivative;  // state derivative
  int dim_action;            // action
  int dim_sensor;            // output (i.e., all sensors)
  int dim_max;               // maximum dimension

  // candidate trajectories
  Trajectory trajectory[kMaxTrajectory];
  int num_trajectory;

  // model derivatives
  ModelDerivatives model_derivative;

  // cost derivatives
  CostDerivatives cost_derivative;

  // backward pass
  iLQGBackwardPass backward_pass;

  // boxQP
  BoxQP boxqp;

  // step sizes
  double linesearch_steps[kMaxTrajectory];

  // best trajectory id
  int winner;

  // settings
  iLQGSettings settings;

  // values
  double action_step;
  double feedback_scaling;
  double improvement;
  double expected;
  double surprise;

  // compute time
  double nominal_compute_time;
  double model_derivative_compute_time;
  double cost_derivative_compute_time;
  double rollouts_compute_time;
  double backward_pass_compute_time;
  double policy_update_compute_time;

  // mutex
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ILQG_OPTIMIZER_H_
