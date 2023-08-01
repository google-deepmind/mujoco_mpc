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

#include "mjpc/estimators/include.h"

namespace mjpc {

const char kEstimatorNames[] =
    "Ground Truth\n"
    "Batch";

// load all available estimators
std::vector<std::unique_ptr<mjpc::Estimator>> LoadEstimators() {
  // planners
  std::vector<std::unique_ptr<mjpc::Estimator>> estimators;

  // add estimators
  estimators.emplace_back(new mjpc::GroundTruth);
  estimators.emplace_back(new mjpc::Batch(1)); // filter mode

  return estimators;
}

}  // namespace mjpc
