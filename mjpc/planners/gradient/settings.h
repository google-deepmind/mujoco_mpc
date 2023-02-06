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

#ifndef MJPC_PLANNERS_GRADIENT_SETTINGS_H_
#define MJPC_PLANNERS_GRADIENT_SETTINGS_H_

namespace mjpc {

// GradientPlanner settings
struct GradientPlannerSettings {
  int max_rollout = 1;           // maximum number of planner iterations
  double min_linesearch_step = 1.0e-8;    // minimum step size for line search
  double fd_tolerance = 1.0e-5;  // finite-difference tolerance
  double fd_mode = 0;  // type of finite difference; 0: one-side, 1: centered
  int action_limits = 1;  // flag
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_GRADIENT_SETTINGS_H_
