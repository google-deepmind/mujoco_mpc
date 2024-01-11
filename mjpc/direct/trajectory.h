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

#ifndef MJPC_DIRECT_TRAJECTORY_H_
#define MJPC_DIRECT_TRAJECTORY_H_

#include <algorithm>
#include <cstring>
#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc {

// maximum length for trajectory length
inline constexpr int kMaxDirectTrajectory = 1024;

// trajectory
template <typename T>
class DirectTrajectory {
 public:
  // constructors
  DirectTrajectory() { Initialize(0, 0); }
  DirectTrajectory(int dim, int length) { Initialize(dim, length); }

  // initialize
  void Initialize(int dim, int length) {
    // set
    dim_ = dim;
    length_ = length;

    // allocate memory
    data_.resize(dim * kMaxDirectTrajectory);

    // reset
    Reset();
  }

  // reset memory
  void Reset() {
    // set head
    head_index_ = 0;

    // zero memory
    std::fill(data_.begin(), data_.end(), 0);
  }

  T* Get(int index) {
    // get mapped index
    int map_index = IndexMap(index);

    // return element
    return data_.data() + dim_ * map_index;
  }

  const T* Get(int index) const {
    // get mapped index
    int map_index = IndexMap(index);

    // return element
    return data_.data() + dim_ * map_index;
  }

  // set element at index
  void Set(const T* element, int index) {
    // get map index
    int map_index = IndexMap(index);

    // get data element
    T* data_element = data_.data() + dim_ * map_index;

    // set element
    std::memcpy(data_element, element, dim_ * sizeof(T));
  }

  // get all data
  T* Data() { return data_.data(); }

  // get head index
  int Head() const { return head_index_; }

  // reset head index
  void ResetHead() { head_index_ = 0; }

  // get element dimension
  int Dimension() const { return dim_; }

  // get trajectory length
  int Length() const { return length_; }

  // set trajectory length
  void SetLength(int length) {
    // set
    length_ = length;

    // potentially fix head index
    Shift(0);
  }

  // shift head_index_
  void Shift(int shift) {
    // check for nonnegative shift
    if (shift < 0) {
      mju_error("shift cannot be negative");
    }

    // compute new head index
    int new_head = head_index_ + shift;

    if (new_head < length_) {  // valid head
      head_index_ = new_head;
    } else {
      head_index_ = new_head % length_;  // corrected head
    }
  }

 private:
  // map index to data_ index
  int IndexMap(int index) const {
    // out of bounds
    if (head_index_ >= length_) mju_error("head index out of bounds!\n");

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

  // index for trajectory head
  int head_index_;

  // dimension of trajectory element
  int dim_;

  // length of trajectory
  int length_;

  // data for trajectory
  std::vector<T> data_;
};

}  // namespace mjpc

#endif  // MJPC_DIRECT_TRAJECTORY_H_
