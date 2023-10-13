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

#ifndef MJPC_TEST_SIMULATION_H_
#define MJPC_TEST_SIMULATION_H_

#include <functional>

#include <mujoco/mujoco.h>

#include "mjpc/direct/trajectory.h"

namespace mjpc {

// convenience class for simulating a system and saving the resulting
// trajectories
class Simulation {
 public:
  // constructor
  Simulation(const mjModel* model, int length);

  // destructor
  ~Simulation() {
    if (data_) mj_deleteData(data_);
    if (model) mj_deleteModel(model);
  }

  // set state
  void SetState(const double* qpos, const double* qvel);

  // rollout
  void Rollout(std::function<void(double* ctrl, double time)> controller);

  // model
  mjModel* model = nullptr;

  // trajectories
  DirectTrajectory<double> qpos;
  DirectTrajectory<double> qvel;
  DirectTrajectory<double> qacc;
  DirectTrajectory<double> ctrl;
  DirectTrajectory<double> time;
  DirectTrajectory<double> sensor;
  DirectTrajectory<double> qfrc_actuator;

 private:
  // data
  mjData* data_ = nullptr;

  // rollout length
  int length_;
};

}  // namespace mjpc

#endif  // MJPC_TEST_SIMULATION_H_
