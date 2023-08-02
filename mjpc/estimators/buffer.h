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

#ifndef MJPC_ESTIMATORS_BUFFER_H_
#define MJPC_ESTIMATORS_BUFFER_H_

#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/estimators/trajectory.h"

namespace mjpc {

// buffer containing: sensor, ctrl, and time data
class Buffer {
 public:
  // constructor
  Buffer() = default;
  Buffer(int dim_sensor, int num_sensor, int dim_ctrl, int max_length) {
    Initialize(dim_sensor, num_sensor, dim_ctrl, max_length);
  };

  // destructor
  ~Buffer() = default;

  // initialize
  void Initialize(int dim_sensor, int num_sensor, int dim_ctrl, int max_length);

  // reset
  void Reset();

  // update
  void Update(const double* sensor, const int* mask, const double* ctrl,
              double time);

  // update mask
  void UpdateMask();

  // print
  void Print();

  // length
  int Length() const;

  // sensor
  EstimatorTrajectory<double> sensor;

  // mask
  EstimatorTrajectory<int> sensor_mask;
  std::vector<int> mask;

  // ctrl
  EstimatorTrajectory<double> ctrl;

  // time
  EstimatorTrajectory<double> time;

 private:
  // max buffer length
  int max_length_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_BUFFER_H_