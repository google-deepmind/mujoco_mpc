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

#include "mjpc/planners/direct/policy.h"

#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void DirectPolicy::Allocate(const mjModel* model, const Task& task,
                            int horizon) {
  // model
  this->model = model;
}

// reset memory to zeros
void DirectPolicy::Reset(int horizon) {}

// set action from policy
void DirectPolicy::Action(double* action, const double* state,
                          double time) const {}

// copy policy
void DirectPolicy::CopyFrom(const DirectPolicy& policy, int horizon) {}

}  // namespace mjpc
