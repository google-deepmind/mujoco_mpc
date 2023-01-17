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

#include "tasks/hand/hand.h"

#include <mujoco/mujoco.h>
#include "task.h"
#include "utilities.h"

namespace mjpc {

// ---------- Residuals for in-hand manipulation task ---------
//   Number of residuals: 5
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): cube angular velocity
//     Residual (4): control
// ------------------------------------------------------------
void Hand::Residual(const double* parameters, const mjModel* model,
                    const mjData* data, double* residual) {
  int counter = 0;
  // ---------- Residual (0) ----------
  // goal position
  double* goal_position = mjpc::SensorByName(model, data, "palm_position");

  // system's position
  double* position = mjpc::SensorByName(model, data, "cube_position");

  // position error
  mju_sub3(residual + counter, position, goal_position);
  counter += 3;

  // ---------- Residual (1) ----------
  // goal orientation
  double* goal_orientation =
      mjpc::SensorByName(model, data, "cube_goal_orientation");

  // system's orientation
  double* orientation = mjpc::SensorByName(model, data, "cube_orientation");
  mju_normalize4(goal_orientation);

  // orientation error
  mju_subQuat(residual + counter, goal_orientation, orientation);
  counter += 3;

  // ---------- Residual (2) ----------
  double* cube_linear_velocity =
      mjpc::SensorByName(model, data, "cube_linear_velocity");
  mju_copy(residual + counter, cube_linear_velocity, 3);
  counter += 3;

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i=0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension "
                "and actual length of residual %d", counter);
  }
}

// ----- Transition for in-hand manipulation task -----
//   If cube is within tolerance or floor ->
//   reset cube into hand.
// -----------------------------------------------
int Hand::Transition(int state, const mjModel* model, mjData* data,
                     Task* task) {
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

  double* cube_lin_vel =
      mjpc::SensorByName(model, data, "cube_linear_velocity");
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

  return state;
}

}  // namespace mjpc
