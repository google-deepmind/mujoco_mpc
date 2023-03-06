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

#include "mjpc/tasks/hand/hand.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string Hand::XmlPath() const {
  return GetModelPath("hand/task.xml");
}
std::string Hand::Name() const { return "Hand"; }

// ---------- Residuals for in-hand manipulation task ---------
//   Number of residuals: 4
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): control
// ------------------------------------------------------------
void Hand::Residual(const mjModel* model, const mjData* data,
                    double* residual) const {
  int counter = 0;
  // ---------- Residual (0) ----------
  // goal position
  double* goal_position = SensorByName(model, data, "palm_position");

  // system's position
  double* position = SensorByName(model, data, "cube_position");

  // position error
  mju_sub3(residual + counter, position, goal_position);
  counter += 3;

  // ---------- Residual (1) ----------
  // goal orientation
  double* goal_orientation = SensorByName(model, data, "cube_goal_orientation");

  // system's orientation
  double* orientation = SensorByName(model, data, "cube_orientation");
  mju_normalize4(goal_orientation);

  // orientation error
  mju_subQuat(residual + counter, goal_orientation, orientation);
  counter += 3;

  // ---------- Residual (2) ----------
  double* cube_linear_velocity =
      SensorByName(model, data, "cube_linear_velocity");
  mju_copy(residual + counter, cube_linear_velocity, 3);
  counter += 3;

  // ---------- Residual (3) ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

// ----- Transition for in-hand manipulation task -----
//   If cube is within tolerance or floor ->
//   reset cube into hand.
// -----------------------------------------------
void Hand::Transition(const mjModel* model, mjData* data) {
  // find cube and floor
  int cube = mj_name2id(model, mjOBJ_GEOM, "cube");
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");
  // look for contacts between the cube and the floor
  bool on_floor = false;
  for (int i=0; i < data->ncon; i++) {
    mjContact* g = data->contact + i;
    if ((g->geom1 == cube && g->geom2 == floor) ||
        (g->geom2 == cube && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  double* cube_lin_vel = SensorByName(model, data, "cube_linear_velocity");
  if (on_floor && mju_norm3(cube_lin_vel) < .001) {
    // reset box pose, adding a little height
    int cube_body = mj_name2id(model, mjOBJ_BODY, "cube");
    if (cube_body != -1) {
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[cube_body]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[cube_body]];
      mju_copy(data->qpos + jnt_qposadr, model->qpos0 + jnt_qposadr, 7);
      mju_zero(data->qvel + jnt_veladr, 6);
    }
    mj_forward(model, data);
  }
}

}  // namespace mjpc
