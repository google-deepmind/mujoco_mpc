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

#include "mjpc/test/load.h"

#include <iostream>
#include <string>
#include <string_view>

// DEEPMIND INTERNAL IMPORT
#include <absl/strings/str_cat.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"

namespace mjpc {

static std::string RelativeTestModelPath(std::string_view path) {
  return absl::StrCat("../../mjpc/test/testdata/", path);
}

// convenience function for paths
mjModel* LoadTestModel(std::string_view path) {
  // filename
  char filename[1024];
  const std::string path_str =
      RelativeTestModelPath(path);
  mujoco::util_mjpc::strcpy_arr(filename, path_str.c_str());

  // load model
  char loadError[1024] = "";
  mjModel* model = mj_loadXML(filename, nullptr, loadError, 1000);
  if (loadError[0]) std::cerr << "load error: " << loadError << '\n';

  return model;
}

}  // namespace mjpc
