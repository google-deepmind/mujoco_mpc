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

// Definitions for tasks built into mujoco_mpc.

#ifndef MJPC_TASKS_TASKS_H_
#define MJPC_TASKS_TASKS_H_

#include "task.h"

namespace mjpc {
inline constexpr int kNumTasks = 12;
extern const TaskDefinition (&kTasks)[kNumTasks];
}  // namespace mjpc

#endif  // MJPC_TASKS_TASKS_H_
