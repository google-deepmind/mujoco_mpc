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

#include "mjpc/planners/gradient/planner.h"
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/ilqs/planner.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/planner.h"

namespace mjpc {
const char kPlannerNames[] =
    "Sampling\n"
    "Gradient\n"
    "iLQG\n"
    "iLQS";

// load all available planners
std::vector<std::unique_ptr<mjpc::Planner>> LoadPlanners() {
  // planners
  std::vector<std::unique_ptr<mjpc::Planner>> planners;

  planners.emplace_back(new mjpc::SamplingPlanner);
  planners.emplace_back(new mjpc::GradientPlanner);
  planners.emplace_back(new mjpc::iLQGPlanner);
  planners.emplace_back(new mjpc::iLQSPlanner);
  return planners;
}

}  // namespace mjpc
