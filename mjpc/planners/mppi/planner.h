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

#ifndef MJPC_PLANNERS_MPPI_PLANNER_H_
#define MJPC_PLANNERS_MPPI_PLANNER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/trajectory.h"

namespace mjpc {

class MPPIPlanner : public Planner {
 public:
  // constructor
  MPPIPlanner() = default;

  // destructor
  ~MPPIPlanner() override = default;

  // ----- methods ----- //

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // set state
  void SetState(const State& state) override;

  // optimize nominal policy using random sampling
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

  // resample nominal policy
  void ResamplePolicy(int horizon);

  // add noise to nominal policy
  void AddNoiseToPolicy(int i);

  // compute candidate trajectories
  void Rollouts(int num_trajectory, int horizon, ThreadPool& pool);

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  SamplingPolicy policy;  // (Guarded by mtx_)
  SamplingPolicy candidate_policy[kMaxTrajectory];
  SamplingPolicy resampled_policy;
  SamplingPolicy previous_policy;

  // scratch
  std::vector<double> parameters_scratch;
  std::vector<double> times_scratch;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];
  Trajectory nominal_trajectory;  // the weighted traj of rollouts

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // rollout parameters
  double timestep_power;

  // ----- noise ----- //
  double noise_exploration_;  // standard deviation for sampling normal: N(0,
                              // std)
  std::vector<double> noise;

  // improvement
  double improvement;

  // flags
  int processed_noise_status;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  // mppi
  double lambda_;                   // the temp of the energy-based model
  std::vector<double> weight_vec;  // MPPI weights
  double denom;                    // sum of weight_vec
  double temp_weight;              // temp variable for storing weights

  int num_trajectory_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_MPPI_PLANNER_H_
