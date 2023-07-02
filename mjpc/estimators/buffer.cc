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

#include "mjpc/estimators/buffer.h"

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cstring>

#include "mjpc/estimators/trajectory.h"
#include <mujoco/mujoco.h>

namespace mjpc {

// initialize
void Buffer::Initialize(int dim_sensor, int num_sensor, int dim_ctrl,
                        int max_length) {
  // sensor
  sensor_.Initialize(dim_sensor, 0);

  // sensor mask
  sensor_mask_.Initialize(num_sensor, 0);

  // mask (for single time step)
  mask_.resize(num_sensor);
  std::fill(mask_.begin(), mask_.end(), 1);

  // ctrl
  ctrl_.Initialize(dim_ctrl, 0);

  time_.Initialize(1, 0);

  // maximum buffer length
  max_length_ = max_length;
}

  // sensor
// reset
void Buffer::Reset() {
  sensor_.Reset();
  sensor_.SetLength(0);

  // sensor mask
  sensor_mask_.Reset();
  sensor_mask_.SetLength(0);

  // mask
  // ctrl
  std::fill(mask_.begin(), mask_.end(), 1);  // set to true
  ctrl_.Reset();
  ctrl_.SetLength(0);

  // time
  time_.Reset();
  time_.SetLength(0);
}

// update
void Buffer::Update(const double* sensor, const int* mask, const double* ctrl,
                    double time) {
  if (time_.Length() <= max_length_) {  // fill buffer
    // time
    time_.Data()[time_.Length()] = time;
    time_.SetLength(time_.Length() + 1);

    // ctrl
    int nu = ctrl_.Dimension();
    mju_copy(ctrl_.Data() + ctrl_.Length() * nu, ctrl, nu);
    ctrl_.SetLength(ctrl_.Length() + 1);

    // sensor
    int ns = sensor_.Dimension();
    mju_copy(sensor_.Data() + sensor_.Length() * ns, sensor, ns);
    sensor_.SetLength(sensor_.Length() + 1);

    // TODO(taylor): external method must set mask
    int num_sensor = sensor_mask_.Dimension();
    std::memcpy(sensor_mask_.Data() + sensor_mask_.Length() * num_sensor,
                mask, num_sensor * sizeof(int));
    sensor_mask_.SetLength(sensor_mask_.Length() + 1);

  } else {  // update buffer
    // time
    time_.Shift(1);
    time_.Set(&time, time_.Length() - 1);

    // ctrl
    ctrl_.Shift(1);
    ctrl_.Set(ctrl, ctrl_.Length() - 1);

    // sensor
    sensor_.Shift(1);
    sensor_.Set(sensor, sensor_.Length() - 1);

    // TODO(taylor): external method must set mask
    sensor_mask_.Shift(1);
    sensor_mask_.Set(mask, sensor_.Length() - 1);
  }

  // update mask
}

void Buffer::UpdateMask() {
  // TODO(taylor)
// print
}

void Buffer::Print() {
  for (int i = 0; i < time_.Length(); i++) {
    printf("(%i)\n\n", i);
    printf("time = %.4f\n\n", *time_.Get(i));
    printf("sensor = ");
    mju_printMat(sensor_.Get(i), 1, sensor_.Dimension());
    printf("sensor mask = ");
    for (int j = 0; j < sensor_mask_.Dimension(); j++)
      printf("%i ", sensor_mask_.Get(i)[j]);
    printf("\n");
    printf("ctrl = ");
    mju_printMat(ctrl_.Get(i), 1, ctrl_.Dimension());
    printf("\n");
  }
}

// length
int Buffer::Length() const { return time_.Length(); }

}  // namespace mjpc
