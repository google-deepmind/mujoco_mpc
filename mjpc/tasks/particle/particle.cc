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

#include "tasks/particle/particle.h"

#include <mujoco/mujoco.h>
#include "utilities.h"

// #include <absl/random/random.h>


namespace mjpc {

// -------- Residuals for particle move-to-goal task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
void Particle::ResidualGoal(const double* parameters, const mjModel* model,
                        const mjData* data, double* residual) {
  // ----- residual (0) ----- //
  double* position = SensorByName(model, data, "position");
  double* goal = SensorByName(model, data, "goal");
  mju_sub(residual, position, goal, model->nq);

  // ----- residual (1) ----- //
  double* velocity = SensorByName(model, data, "velocity");
  mju_copy(residual + 2, velocity, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 4, data->ctrl, model->nu);
}

// -------- Residuals for particle move-to-goal task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
void Particle::ResidualTrack(const double* parameters, const mjModel* model,
                        const mjData* data, double* residual) {

  // ----- set goal ----- //
  // circle
  double period = 10.0;
  double radius = 0.25;

  // goal
  double goal[2];
  double angle = 2.0 * 3.141592 * data->time / period;
  goal[0] = radius * mju_cos(angle);
  goal[1] = radius * mju_sin(angle);
  
  // ----- residual (0) ----- //
  double* position = SensorByName(model, data, "position");
  mju_sub(residual, position, goal, model->nq);

  // ----- residual (1) ----- //
  double* velocity = SensorByName(model, data, "velocity");
  mju_copy(residual + 2, velocity, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 4, data->ctrl, model->nu);
}

// -------- Transition for particle track task --------
// Set goal to next.
// -----------------------------------------------
int Particle::Transition(int state, const mjModel* model, mjData* data) {
  int new_state = state;

  // circle
  double period = 10.0;
  double radius = 0.25;

  // position
  double angle = 2.0 * 3.141592 * data->time / period;
  double x = radius * mju_cos(angle);
  double y = radius * mju_sin(angle);

  // update mocap position
  data->mocap_pos[0] = x;
  data->mocap_pos[1] = y;

  return new_state;
}

}  // namespace mjpc
