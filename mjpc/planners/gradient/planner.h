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

#ifndef MJPC_PLANNERS_GRADIENT_PLANNER_H_
#define MJPC_PLANNERS_GRADIENT_PLANNER_H_

#include <memory>
#include <shared_mutex>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/planners/cost_derivatives.h"
#include "mjpc/planners/gradient/gradient.h"
#include "mjpc/planners/gradient/policy.h"
#include "mjpc/planners/gradient/settings.h"
#include "mjpc/planners/gradient/spline_mapping.h"
#include "mjpc/planners/model_derivatives.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// first-order gradient descent planner
class GradientPlanner : public Planner {
 public:
  // constructor
  GradientPlanner() {
    mappings.emplace_back(new ZeroSplineMapping);
    mappings.emplace_back(new LinearSplineMapping);
    mappings.emplace_back(new CubicSplineMapping);
  }

  // ----- methods ----- //

  // initialize planner settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // set state
  void SetState(const State& state) override;

  // optimize nominal policy via gradient descent
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // compute action from policy
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

  // resample nominal policy for current time
  void ResamplePolicy(int horizon);

  // compute candidate trajectories
  void Rollouts(int horizon, ThreadPool& pool);

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize candidate traces in GUI
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // return number of parameters optimized by planner
  int NumParameters() override {
    return policy.num_spline_points * policy.model->nu;
  };

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  GradientPolicy policy;
  GradientPolicy previous_policy;
  GradientPolicy candidate_policy[kMaxTrajectory];

  // scratch
  std::vector<double> parameters_scratch;
  std::vector<double> times_scratch;

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

  // gradient descent
  Gradient gradient;

  // spline mapping
  std::vector<std::unique_ptr<SplineMapping>> mappings;

  // step sizes
  double linesearch_steps[kMaxTrajectory];

  // best trajectory id
  int winner;

  // settings
  GradientPlannerSettings settings;

  // values
  double action_step;
  double expected;
  double improvement;
  double surprise;

  // compute time
  double nominal_compute_time;
  double model_derivative_compute_time;
  double cost_derivative_compute_time;
  double rollouts_compute_time;
  double gradient_compute_time;
  double policy_update_compute_time;

 private:
  mutable std::shared_mutex mtx_;
  int derivative_skip_ = 0;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_GRADIENT_PLANNER_H_
