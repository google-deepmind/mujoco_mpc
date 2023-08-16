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

#include <mujoco/mujoco.h>

#include <stdio.h>
#include <algorithm>
#include <vector>
#include <cstring>

#include "mjpc/estimators/trajectory.h"

namespace mjpc {

// initialize
void Buffer::Initialize(int dim_sensor, int num_sensor, int dim_ctrl,
                        int max_length) {
  // sensor
  sensor.Initialize(dim_sensor, 0);

  // sensor mask
  sensor_mask.Initialize(num_sensor, 0);

  // mask (for single time step)
  mask.resize(num_sensor);
  std::fill(mask.begin(), mask.end(), 1);

  // ctrl
  ctrl.Initialize(dim_ctrl, 0);

  time.Initialize(1, 0);

  // maximum buffer length
  max_length_ = max_length;
}

  // sensor
// reset
void Buffer::Reset() {
  sensor.Reset();
  sensor.SetLength(0);

  // sensor mask
  sensor_mask.Reset();
  sensor_mask.SetLength(0);

  // mask
  // ctrl
  std::fill(mask.begin(), mask.end(), 1);  // set to true
  ctrl.Reset();
  ctrl.SetLength(0);

  // time
  time.Reset();
  time.SetLength(0);
}

// update
void Buffer::Update(const double* sensor, const int* mask, const double* ctrl,
                    double time) {
  if (this->time.Length() <= max_length_) {  // fill buffer
    // time
    this->time.Data()[this->time.Length()] = time;
    this->time.SetLength(this->time.Length() + 1);

    // ctrl
    int nu = this->ctrl.Dimension();
    mju_copy(this->ctrl.Data() + this->ctrl.Length() * nu, ctrl, nu);
    this->ctrl.SetLength(this->ctrl.Length() + 1);

    // sensor
    int ns = this->sensor.Dimension();
    mju_copy(this->sensor.Data() + this->sensor.Length() * ns, sensor, ns);
    this->sensor.SetLength(this->sensor.Length() + 1);

    // TODO(taylor): external method must set mask
    int num_sensor = sensor_mask.Dimension();
    std::memcpy(sensor_mask.Data() + sensor_mask.Length() * num_sensor,
                mask, num_sensor * sizeof(int));
    sensor_mask.SetLength(sensor_mask.Length() + 1);

  } else {  // update buffer
    // time
    this->time.Shift(1);
    this->time.Set(&time, this->time.Length() - 1);

    // ctrl
    this->ctrl.Shift(1);
    this->ctrl.Set(ctrl, this->ctrl.Length() - 1);

    // sensor
    this->sensor.Shift(1);
    this->sensor.Set(sensor, this->sensor.Length() - 1);

    // TODO(taylor): external method must set mask
    sensor_mask.Shift(1);
    sensor_mask.Set(mask, this->sensor.Length() - 1);
  }

  // update mask
}

void Buffer::UpdateMask() {
  // TODO(taylor)
// print
}

void Buffer::Print() {
  for (int i = 0; i < time.Length(); i++) {
    printf("(%i)\n\n", i);
    printf("time = %.4f\n\n", *time.Get(i));
    printf("sensor = ");
    mju_printMat(sensor.Get(i), 1, sensor.Dimension());
    printf("sensor mask = ");
    for (int j = 0; j < sensor_mask.Dimension(); j++)
      printf("%i ", sensor_mask.Get(i)[j]);
    printf("\n");
    printf("ctrl = ");
    mju_printMat(ctrl.Get(i), 1, ctrl.Dimension());
    printf("\n");
  }
}

// length
int Buffer::Length() const { return time.Length(); }

}  // namespace mjpc
