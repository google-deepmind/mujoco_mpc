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

#include "mjpc/planners/include.h"

#include <memory>
#include <vector>

#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/planners/gradient/planner.h"
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/ilqs/planner.h"
#include "mjpc/planners/mppi/planner.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/robust/robust_planner.h"
#include "mjpc/planners/sampling/planner.h"

namespace mjpc {
const char kPlannerNames[] =
    "Sampling\n"
    "Gradient\n"
    "iLQG\n"
    "iLQS\n"
    "Robust Sampling\n"
    "Cross Entropy\n"
    "MPPI";

// load all available planners
std::vector<std::unique_ptr<mjpc::Planner>> LoadPlanners() {
  // planners
  std::vector<std::unique_ptr<mjpc::Planner>> planners;

  planners.emplace_back(new mjpc::SamplingPlanner);
  planners.emplace_back(new mjpc::GradientPlanner);
  planners.emplace_back(new mjpc::iLQGPlanner);
  planners.emplace_back(new mjpc::iLQSPlanner);
  planners.emplace_back(
      new RobustPlanner(std::make_unique<mjpc::SamplingPlanner>()));
  planners.emplace_back(new mjpc::CrossEntropyPlanner);
  planners.emplace_back(new mjpc::MPPIPlanner);
  return planners;
}

}  // namespace mjpc
