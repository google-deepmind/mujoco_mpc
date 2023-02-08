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

#ifndef MJPC_PLANNERS_ILQG_POLICY_H_
#define MJPC_PLANNERS_ILQG_POLICY_H_

#include <vector>

#include "mjpc/planners/policy.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// iLQG policy
class iLQGPolicy : public Policy {
 public:
  // constructor
  iLQGPolicy() = default;

  // destructor
  ~iLQGPolicy() override = default;

  // allocate memory
  void Allocate(const mjModel* model, const Task& task, int horizon) override;

  // reset memory to zeros
  void Reset(int horizon) override;

  // set action from policy
  void Action(double* action, const double* state, double time) const override;

  // copy policy
  void CopyFrom(const iLQGPolicy& policy, int horizon);

 public:
  // ----- members ----- //
  const mjModel* model;

  Trajectory trajectory;              // reference trajectory
  std::vector<double> feedback_gain;  // (T * dim_action * dim_state_derivative)
  std::vector<double> action_improvement;  // (T * dim_action)

  // scratch space
  mutable std::vector<double> state_scratch;       // dim_state
  mutable std::vector<double> action_scratch;      // dim_action

  // interpolation
  mutable std::vector<double> feedback_gain_scratch;
  mutable std::vector<double> state_interp;
  int representation;
  double feedback_scaling;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ILQG_POLICY_H_
