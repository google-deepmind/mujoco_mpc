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

#include "mjpc/planners/sampling/policy.h"

#include <absl/log/check.h>
#include <absl/types/span.h>
#include <mujoco/mujoco.h>
#include "mjpc/spline/spline.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

using mjpc::spline::TimeSpline;

// allocate memory
void SamplingPolicy::Allocate(const mjModel* model, const Task& task,
                              int horizon) {
  // model
  this->model = model;

  // spline points
  num_spline_points = GetNumberOrDefault(kMaxTrajectoryHorizon, model,
                                         "sampling_spline_points");

  plan = TimeSpline(/*dim=*/model->nu);
  plan.Reserve(num_spline_points);
}

// reset memory to zeros
void SamplingPolicy::Reset(int horizon, const double* initial_repeated_action) {
  plan.Clear();
  if (initial_repeated_action != nullptr) {
    plan.AddNode(0, absl::MakeConstSpan(initial_repeated_action, model->nu));
  }
}

// set action from policy
void SamplingPolicy::Action(double* action, const double* state,
                            double time) const {
  CHECK(action != nullptr);
  plan.Sample(time, absl::MakeSpan(action, model->nu));

  // Clamp controls
  Clamp(action, model->actuator_ctrlrange, model->nu);
}

// copy policy
void SamplingPolicy::CopyFrom(const SamplingPolicy& policy, int horizon) {
  this->plan = policy.plan;
  num_spline_points = policy.num_spline_points;
}

// copy parameters
void SamplingPolicy::SetPlan(const TimeSpline& plan) {
  this->plan = plan;
}

}  // namespace mjpc
