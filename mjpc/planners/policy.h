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

#ifndef MJPC_PLANNERS_POLICY_H_
#define MJPC_PLANNERS_POLICY_H_

#include <mujoco/mjmodel.h>
#include "mjpc/task.h"

namespace mjpc {

// type of interpolation
enum PolicyRepresentation : int {
  kZeroSpline,
  kLinearSpline,
  kCubicSpline,
};

// virtual policy
class Policy {
 public:
  // destructor
  virtual ~Policy() = default;

  // allocate memory
  virtual void Allocate(const mjModel* model, const Task& task,
                        int horizon) = 0;

  // reset memory to zeros
  virtual void Reset(int horizon) = 0;

  // set action from policy
  virtual void Action(double* action, const double* state,
                      double time) const = 0;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_POLICY_H_
