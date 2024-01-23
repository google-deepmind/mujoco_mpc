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

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <string>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string Hand::XmlPath() const { return GetModelPath("hand/task.xml"); }
std::string Hand::Name() const { return "Hand"; }

// ---------- Residuals for in-hand manipulation task ---------
//   Number of residuals: 6
//     Residual (0): cube_position - palm_position
//     Residual (1): cube_orientation - cube_goal_orientation
//     Residual (2): cube linear velocity
//     Residual (3): control
//     Residual (4): hand configuration - nominal hand configuration
//     Residual (5): hand joint velocity
// ------------------------------------------------------------
void Hand::ResidualFn::Residual(const mjModel* model, const mjData* data,
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

  // ---------- Residual (4) ----------
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, 26);
  counter += 26;

  // ---------- Residual (5) ----------
  mju_copy(residual + counter, data->qvel + 6, 26);
  counter += 26;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

// ----- Transition for in-hand manipulation task -----
//   If cube is within tolerance or floor ->
//   reset cube into hand.
// -----------------------------------------------
void Hand::TransitionLocked(mjModel* model, mjData* data) {
  // find cube and floor
  int cube = mj_name2id(model, mjOBJ_GEOM, "cube");
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");
  // look for contacts between the cube and the floor
  bool on_floor = false;
  for (int i = 0; i < data->ncon; i++) {
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
    mutex_.unlock();          // step calls sensor that calls Residual.
    mj_forward(model, data);  // mj_step1 would suffice, we just need contact
    mutex_.lock();
  }

  // update mocap position
  std::vector<double> pos_cube = pos_cube_;
  std::vector<double> quat_cube = quat_cube_;
  data->mocap_pos[0] = pos_cube[0];
  data->mocap_pos[1] = pos_cube[1];
  data->mocap_pos[2] = pos_cube[2];
  data->mocap_quat[0] = quat_cube[0];
  data->mocap_quat[1] = quat_cube[1];
  data->mocap_quat[2] = quat_cube[2];
  data->mocap_quat[3] = quat_cube[3];
}

void Hand::ModifyState(const mjModel* model, State* state) {
  // sampling token
  absl::BitGen gen_;

  // std from GUI
  double std_rot = parameters[0];    // concentration parameter ("inverse var")
  double std_pos = parameters[1];    // uniform stdev for position noise
  double bias_posx = parameters[2];  // bias for position noise
  double bias_posy = parameters[3];  // bias for position noise
  double bias_posz = parameters[4];  // bias for position noise

  // current state
  const std::vector<double>& s = state->state();

  // add quaternion noise
  std::vector<double> dv = {0.0, 0.0, 0.0};  // rotational velocity noise
  dv[0] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  dv[1] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  dv[2] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  std::vector<double> quat_cube = {s[7], s[8], s[9], s[10]};  // quat cube state
  mju_quatIntegrate(quat_cube.data(), dv.data(), 1.0);        // update the quat
  mju_normalize4(quat_cube.data());  // normalize the quat for numerics

  // add position noise
  std::vector<double> dp = {bias_posx, bias_posy,
                            bias_posz};  // translational velocity noise
  dp[0] += absl::Gaussian<double>(gen_, 0.0, std_pos);
  dp[1] += absl::Gaussian<double>(gen_, 0.0, std_pos);
  dp[2] += absl::Gaussian<double>(gen_, 0.0, std_pos);
  std::vector<double> pos_cube = {s[4], s[5], s[6]};  // position cube state
  mju_addTo3(pos_cube.data(), dp.data());             // update the pos

  // set state
  std::vector<double> qpos(model->nq);
  mju_copy(qpos.data(), s.data(), model->nq);
  mju_copy(qpos.data() + 7, quat_cube.data(), 4);
  mju_copy(qpos.data() + 4, pos_cube.data(), 3);
  state->SetPosition(model, qpos.data());

  // update cube mocap state
  mju_copy(pos_cube_.data(), pos_cube.data(), 3);
  mju_copy(quat_cube_.data(), quat_cube.data(), 4);
}

}  // namespace mjpc
