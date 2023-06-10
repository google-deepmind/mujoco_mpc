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

#include "mjpc/estimators/trajectory.h"

#include <mujoco/mujoco.h>

#include <cstring>
#include <vector>

namespace mjpc {
// initialize
void EstimatorTrajectory::Initialize(int dim, int length) {
  // set
  dim_ = dim;
  length_ = length;

  // allocate memory
  data_.resize(dim * MAX_TRAJECTORY);

  // reset
  Reset();
}

// reset memory
void EstimatorTrajectory::Reset() {
  // set head
  head_index_ = 0;

  // zero memory
  std::fill(data_.begin(), data_.end(), 0.0);
}

// get element at index
double* EstimatorTrajectory::Get(int index) {
  // get mapped index
  int map_index = IndexMap(index);

  // return element
  return data_.data() + dim_ * map_index;
}

const double* EstimatorTrajectory::Get(int index) const {
  // get mapped index
  int map_index = IndexMap(index);

  // return element
  return data_.data() + dim_ * map_index;
}

// set element at index
void EstimatorTrajectory::Set(const double* element, int index) {
  // get map index
  int map_index = IndexMap(index);

  // get data element
  double* data_element = data_.data() + dim_ * map_index;

  // set element
  // mju_copy(data_element, element, dim_);
  std::memcpy(data_element, element, dim_ * sizeof(double));
}

// get all data
double* EstimatorTrajectory::Data() { return data_.data(); }

// map index to data_ index
// TODO(taylor): index beyond 2 * length
int EstimatorTrajectory::IndexMap(int index) const {
  // out of bounds
  if (head_index_ >= length_)
    mju_error("trajectory.head_index_ out of bounds!\n");

  // if synced
  if (head_index_ == 0) return index;

  // not synced
  int map = head_index_ + index;

  if (map < length_) {  // valid map
    return map;
  } else {  // corrected map
    return map % length_;
  }
}

// shift head_index_
// TODO(taylor): shift beyond 2 * length
void EstimatorTrajectory::ShiftHeadIndex(int shift) {
  // compute new head index
  int new_head = head_index_ + shift;

  if (new_head < length_) {  // valid head
    head_index_ = new_head;
  } else {
    head_index_ = new_head % length_;  // corrected head
  }
}

}  // namespace mjpc
