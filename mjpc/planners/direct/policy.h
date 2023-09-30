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

#ifndef MJPC_PLANNERS_DIRECT_POLICY_H_
#define MJPC_PLANNERS_DIRECT_POLICY_H_

#include <vector>

#include "mjpc/planners/policy.h"
#include "mjpc/task.h"

namespace mjpc {

// iLQG policy
class DirectPolicy : public Policy {
 public:
  // constructor
  DirectPolicy() = default;

  // destructor
  ~DirectPolicy() override = default;

  // allocate memory
  void Allocate(const mjModel* model, const Task& task, int horizon) override;

  // reset memory to zeros
  void Reset(int horizon) override;

  // set action from policy
  // if state == nullptr, return the nominal action without a feedback term
  void Action(double* action, const double* state, double time) const override;

  // copy policy
  void CopyFrom(const DirectPolicy& policy, int horizon);

 public:
  // ----- members ----- //
  const mjModel* model;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_DIRECT_POLICY_H_
