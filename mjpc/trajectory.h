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

#ifndef MJPC_TRAJECTORY_H_
#define MJPC_TRAJECTORY_H_

#include <functional>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

// maximum trajectory length
inline constexpr int kMaxTrajectoryHorizon = 512;

// time series of states, actions, costs, residual, times, parameters, noise,
// traces
class Trajectory {
 public:
  // constructor
  Trajectory() = default;
  Trajectory(const Trajectory& other) = default;
  Trajectory& operator=(const Trajectory& other) = default;

  // ----- methods -----//

  // initialize trajectory dimensions
  void Initialize(int dim_state, int dim_action, int dim_residual,
                  int num_trace, int horizon);

  // allocate memory
  void Allocate(int T);

  // reset memory to zeros
  void Reset(int T);

  // simulate model forward in time with continuous-time indexed policy
  void Rollout(
      std::function<void(double* action, const double* state, double time)>
          policy,
      const Task* task, const mjModel* model, mjData* data, const double* state,
      double time, const double* mocap, const double* userdata, int steps);

  // simulate model forward in time with discrete-time indexed policy
  void RolloutDiscrete(
      std::function<void(double* action, const double* state, int index)>
          policy,
      const Task* task, const mjModel* model, mjData* data, const double* state,
      double time, const double* mocap, const double* userdata, int steps);

  // ----- members ----- //
  int horizon;                   // trajectory length
  int dim_state;                 // states dimension
  int dim_action;                // actions dimension
  int dim_residual;              // residual dimension
  int dim_trace;                 // traces dimension
  std::vector<double> states;    // (horizon   x nq + nv + na)
  std::vector<double> actions;   // (horizon-1 x num_action)
  std::vector<double> times;     // horizon
  std::vector<double> residual;  // (horizon x num_residual)
  std::vector<double> costs;     // horizon
  std::vector<double> trace;     // (horizon   x 3)
  double total_return;           // (1)
  bool failure;                  // true if last rollout had a warning

 private:
  // calculates total_return and costs
  void UpdateReturn(const Task* task);
};

}  // namespace mjpc

#endif  // MJPC_TRAJECTORY_H_
