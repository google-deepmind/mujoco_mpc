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

#ifndef MJPC_ESTIMATORS_INCLUDE_H_
#define MJPC_ESTIMATORS_INCLUDE_H_

#include <memory>
#include <vector>

#include "mjpc/estimators/estimator.h"
#include "mjpc/estimators/kalman.h"
#include "mjpc/estimators/unscented.h"

namespace mjpc {

// Estimator names, separated by '\n'.
extern const char kEstimatorNames[];

// Loads all available estimators
std::vector<std::unique_ptr<mjpc::Estimator>> LoadEstimators();

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_INCLUDE_H_
