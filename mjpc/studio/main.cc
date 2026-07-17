// Copyright 2026 DeepMind Technologies Limited
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
#include <absl/flags/parse.h>
#include "third_party/mujoco/src/experimental/studio/launcher.h"
#include "third_party/mujoco_mpc/mjpc/studio/plugin.h"

// Define model_file flag here as it is not defined in launcher library.
ABSL_FLAG(std::string, model_file, "", "MuJoCo model file.");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);

  mujoco::studio::LauncherConfig config;
  config.title = "MuJoCo MPC Studio";

  std::string model_file = absl::GetFlag(FLAGS_model_file);
  if (model_file.empty() && argc > 1 && argv[1][0] != '-') model_file = argv[1];

  config.model_file = model_file;
  mjpc::studio::PrepareToLaunch(config);

  return mujoco::studio::LaunchStudio(argc, argv, config);
}
