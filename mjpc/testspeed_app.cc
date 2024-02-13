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

#include <string>

#include <absl/flags/parse.h>
#include <absl/flags/flag.h>

#include "mjpc/testspeed.h"
#include "mjpc/utilities.h"

ABSL_FLAG(std::string, task, "Cube Solving", "Which model to load on startup.");
ABSL_FLAG(int, planner_thread, mjpc::NumAvailableHardwareThreads() - 5,
          "Number of planner threads to use.");
ABSL_FLAG(int, steps_per_planning_iteration, 4,
          "How many physics steps to take between planning iterations.");
ABSL_FLAG(double, total_time, 10, "Total time to simulate (seconds).");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  std::string task_name = absl::GetFlag(FLAGS_task);
  int planner_thread_count = absl::GetFlag(FLAGS_planner_thread);
  int steps_per_planning_iteration =
      absl::GetFlag(FLAGS_steps_per_planning_iteration);
  double total_time = absl::GetFlag(FLAGS_total_time);
  double cost =
      mjpc::SynchronousPlanningCost(task_name, planner_thread_count,
                                    steps_per_planning_iteration, total_time);
  if (cost < 0) {
    return -1;
  }
}
