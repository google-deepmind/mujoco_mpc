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

#ifndef MUJOCO_MPC_DM_CONTROL_UTILS_REWARDS_H
#define MUJOCO_MPC_DM_CONTROL_UTILS_REWARDS_H

#include <string>
#include <utility>

//
// this file contains the reimplementation of some of the utility functions from
// dm_control the original implementation is from Google DeepMind and can be
// found here:
// https://github.com/google-deepmind/dm_control/tree/main/dm_control/utils/rewards.py
//

double sigmoid(double x, double value_at_1, std::string sigmoid_type);

double tolerance(double x, std::pair<double, double> bounds = {0.0, 0.0},
                 double margin = 0.0, std::string sigmoid_type = "gaussian",
                 double value_at_margin = 0.1);

#endif  // MUJOCO_MPC_DM_CONTROL_UTILS_REWARDS_H
