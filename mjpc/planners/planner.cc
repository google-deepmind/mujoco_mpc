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

#include "mjpc/planners/planner.h"

#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {
void Planner::ResizeMjData(const mjModel* model, int num_threads) {
  int new_size = std::max(1, num_threads);
  if (data_.size() > new_size) {
    data_.erase(data_.begin() + new_size, data_.end());
  } else {
    data_.reserve(new_size);
    while (data_.size() < new_size) {
      data_.push_back(MakeUniqueMjData(mj_makeData(model)));
    }
  }
}
}  // namespace mjpc
