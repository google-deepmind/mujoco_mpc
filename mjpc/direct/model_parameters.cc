// Copyright 2023 DeepMind Technologies Limited
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

#include "mjpc/direct/model_parameters.h"

#include <memory>
#include <vector>

namespace mjpc {

// Loads all available ModelParameters
std::vector<std::unique_ptr<mjpc::ModelParameters>> LoadModelParameters() {
  // model parameters
  std::vector<std::unique_ptr<mjpc::ModelParameters>> model_parameters;

  // add model parameters
  model_parameters.emplace_back(new mjpc::Particle1DDampedParameters());
  model_parameters.emplace_back(new mjpc::Particle1DFramePosParameters());

  return model_parameters;
}

}  // namespace mjpc
