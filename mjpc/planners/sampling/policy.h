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

#ifndef MJPC_PLANNERS_SAMPLING_POLICY_H_
#define MJPC_PLANNERS_SAMPLING_POLICY_H_

#include <vector>

#include <absl/random/distributions.h>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/planners/policy.h"

namespace mjpc {

// policy for sampling planner
class SamplingPolicy : public Policy {
 public:
  // constructor
  SamplingPolicy() = default;

  // destructor
  ~SamplingPolicy() override = default;

  // ----- methods ----- //

  // allocate memory
  void Allocate(const mjModel* model, const Task& task, int horizon) override;

  // reset memory to zeros
  void Reset(int horizon) override;

  // set action from policy
  void Action(double* action, const double* state, double time) const override;

  // copy policy
  void CopyFrom(const SamplingPolicy& policy, int horizon);

  // copy parameters
  void CopyParametersFrom(const std::vector<double>& src_parameters,
                          const std::vector<double>& src_times);

  // ----- members ----- //
  const mjModel* model;
  std::vector<double> parameters;
  std::vector<double> times;
  int num_parameters;
  int num_spline_points;
  PolicyRepresentation representation;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_SAMPLING_POLICY_H_
