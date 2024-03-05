// Copyright 2021 DeepMind Technologies Limited
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

#include <iostream>
#include <string>
#include <vector>
#include <absl/flags/parse.h>

#include <absl/flags/flag.h>
#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include "mjpc/app.h"
#include "mjpc/tasks/tasks.h"

ABSL_FLAG(std::string, task, "Quadruped Flat",
          "Which model to load on startup.");

// machinery for replacing command line error by a macOS dialog box
// when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char* title, const char* msg);
static const char* rosetta_error_msg = nullptr;
__attribute__((used, visibility("default")))
extern "C" void _mj_rosettaError(const char* msg) {
  rosetta_error_msg = msg;
}
#endif

// run event loop
int main(int argc, char** argv) {
  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg) {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif
  absl::ParseCommandLine(argc, argv);

  std::string task_name = absl::GetFlag(FLAGS_task);
  auto tasks = mjpc::GetTasks();
  int task_id = -1;
  for (int i = 0; i < tasks.size(); i++) {
    if (absl::EqualsIgnoreCase(task_name, tasks[i]->Name())) {
      task_id = i;
      break;
    }
  }
  if (task_id == -1) {
    std::cerr << "Invalid --task flag: '" << task_name
              << "'. Valid values:\n";
    for (int i = 0; i < tasks.size(); i++) {
      std::cerr << "  " << tasks[i]->Name() << "\n";
    }
    mju_error("Invalid --task flag.");
  }

  mjpc::StartApp(tasks, 11);  // start with quadruped flat
  return 0;
}
