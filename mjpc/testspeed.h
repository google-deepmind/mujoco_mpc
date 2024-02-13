// Copyright 2024 DeepMind Technologies Limited
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

#ifndef MJPC_MJPC_TESTSPEED_H_
#define MJPC_MJPC_TESTSPEED_H_

#include <string>

namespace mjpc {
double SynchronousPlanningCost(std::string task_name, int planner_thread_count,
                               int steps_per_planning_iteration,
                               double total_time);
}  // namespace mjpc

#endif  // MJPC_MJPC_TESTSPEED_H_
