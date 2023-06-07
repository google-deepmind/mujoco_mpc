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

#ifndef MJPC_ESTIMATORS_TRAJECTORY_H_
#define MJPC_ESTIMATORS_TRAJECTORY_H_

#include <vector>

namespace mjpc {

const int MAX_TRAJECTORY = 128;

// trajectory
// TODO(taylor): template for data_ type
class Trajectory {
 public:
  // constructor
  Trajectory() = default;
  Trajectory(int dim, int length) { Initialize(dim, length); };

  // destructor
  virtual ~Trajectory() = default;

  // initialize
  void Initialize(int dim, int length);

  // reset memory
  void Reset();

  // get element at index
  double* Get(int index);
  const double* Get(int index) const;

  // set element at index
  void Set(const double* element, int index);

  // get all data
  double* Data();

  // map index to data_ index
  int IndexMap(int index) const;

  // shift head_index_ 
  void ShiftHeadIndex(int shift);

  // index for trajectory head
  int head_index_;

  // dimension of trajectory element
  int dim_;

  // length of trajectory
  int length_;

  // data for trajectory
  std::vector<double> data_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_TRAJECTORY_H_
