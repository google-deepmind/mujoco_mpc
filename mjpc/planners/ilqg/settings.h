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

#ifndef MJPC_PLANNERS_ILQG_SETTINGS_H_
#define MJPC_PLANNERS_ILQG_SETTINGS_H_

namespace mjpc {

// iLQG settings
struct iLQGSettings {
  double min_linesearch_step = 1.0e-3;  // minimum step size for line search
  double fd_tolerance = 1.0e-6;   // finite difference tolerance
  double fd_mode = 0;  // type of finite difference; 0: one-sided, 1: centered
  double min_regularization = 1.0e-6;  // minimum regularization value
  double max_regularization = 1.0e6;   // maximum regularization value
  int regularization_type = 0;  // 0: control; 1: feedback; 2: value; 3: none
  int max_regularization_iterations =
      5;  // maximum number of regularization updates per iteration
  int action_limits = 1;  // flag
  int nominal_feedback_scaling = 1; // flag
  int verbose = 0;        // print optimizer info
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ILQG_SETTINGS_H_
