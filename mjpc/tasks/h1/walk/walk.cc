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

#include "mjpc/tasks/h1/walk/walk.h"

#include <iostream>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include <mujoco/mujoco.h>

namespace mjpc::h1 {
std::string Walk::XmlPath() const { return GetModelPath("h1/walk/task.xml"); }
std::string Walk::Name() const { return "H1 Walk"; }

// ------------------ Residuals for humanoid walk task ------------
//   Number of residuals:
//     Residual (0): torso height
//     Residual (1): pelvis-feet aligment
//     Residual (2): balance
//     Residual (3): upright
//     Residual (4): upper posture
//     Residual (5): lower posture
//     Residual (6): face towards goal
//     Residual (7): walk towards goal
//     Residual (8): move feet
//     Residual (9): control
//     Residual (10): feet distance
//     Residual (11): leg cross
//     Residual (12): slippage
//   Number of parameters:
//     Parameter (0): torso height goal
//     Parameter (1): speed goal
//     Parameter (2): feet distance goal
//     Parameter (3): balance speed
// ----------------------------------------------------------------
void Walk::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                double *residual) const {
  int counter = 0;

  // ----- torso height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[counter++] = torso_height - parameters_[0];

  // ----- pelvis / feet ----- //
  double *foot_right = SensorByName(model, data, "foot_right");
  double *foot_left = SensorByName(model, data, "foot_left");
  double pelvis_height = SensorByName(model, data, "pelvis_position")[2];
  residual[counter++] =
      0.5 * (foot_left[2] + foot_right[2]) - pelvis_height - 0.2;

  // ----- balance ----- //
  // capture point
  double *subcom = SensorByName(model, data, "torso_subcom");
  double *subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[3];
  mju_addScl(capture_point, subcom, subcomvel, parameters_[3], 3);
  capture_point[2] = 1.0e-3;

  // project onto line segment

  double axis[3];
  double center[3];
  double vec[3];
  double pcp[3];
  mju_sub3(axis, foot_right, foot_left);
  axis[2] = 1.0e-3;
  double length = 0.5 * mju_normalize3(axis) - 0.05;
  mju_add3(center, foot_right, foot_left);
  mju_scl3(center, center, 0.5);
  mju_sub3(vec, capture_point, center);

  // project onto axis
  double t = mju_dot3(vec, axis);

  // clamp
  t = mju_max(-length, mju_min(length, t));
  mju_scl3(vec, axis, t);
  mju_add3(pcp, vec, center);
  pcp[2] = 1.0e-3;

  // is standing
  double standing =
      torso_height / mju_sqrt(torso_height * torso_height + 0.45 * 0.45) - 0.4;

  mju_sub(&residual[counter], capture_point, pcp, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);

  counter += 2;

  // ----- upright ----- //
  double *torso_up = SensorByName(model, data, "torso_up");
  double *pelvis_up = SensorByName(model, data, "pelvis_up");
  double *foot_right_up = SensorByName(model, data, "foot_right_up");
  double *foot_left_up = SensorByName(model, data, "foot_left_up");
  double z_ref[3] = {0.0, 0.0, 1.0};

  // torso
  residual[counter++] = torso_up[2] - 1.0;

  // pelvis
  residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);

  // right foot
  mju_sub3(&residual[counter], foot_right_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;

  mju_sub3(&residual[counter], foot_left_up, z_ref);
  mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
  counter += 3;
  // ----- posture up ----- //
  mju_copy(&residual[counter], data->qpos + 17,
           model->nq - 17); // First 7 are freejoint coord, the other 10 are
                            // lower body joints
  counter += model->nq - 17;
  // ----- posture down ----- //
  mju_copy(&residual[counter], data->qpos + 7,
           model->nq - 16); // First 7 are freejoint coord, the other 10 are
                            // lower body joints
  counter += model->nq - 16;
  // ----- walk ----- //
  double *torso_forward = SensorByName(model, data, "torso_forward");
  double *pelvis_forward = SensorByName(model, data, "pelvis_forward");
  double *foot_right_forward = SensorByName(model, data, "foot_right_forward");
  double *foot_left_forward = SensorByName(model, data, "foot_left_forward");

  double forward_target[2];
  double forward[2];
  mju_copy(forward, torso_forward, 2);
  mju_addTo(forward, pelvis_forward, 2);
  mju_addTo(forward, foot_right_forward, 2);
  mju_addTo(forward, foot_left_forward, 2);
  mju_normalize(forward, 2);

  double *goal_point = SensorByName(model, data, "goal");
  double *torso_position = SensorByName(model, data, "torso_position");
  mju_sub(forward_target, goal_point, torso_position, 2);
  double goal_distance = mju_normalize(forward_target, 2);
  // A function of the distance to the goal used to disable goal tracking when the goal is too close.
  // To do this, we use a tanh function that tends to 0 when the goal is less than 30cm away and 1 otherwise.
  double goal_distance_factor = std::tanh((goal_distance - 0.3) / 0.01) / 2.0 + 0.5;
  double com_vel[2];
  mju_copy(com_vel, subcomvel, 2); // subcomvel is the velocity of the robot's CoM

  // Extract the goal forward direction from the goal point
  double *goal_forward = SensorByName(model, data, "goal_forward");

  // face goal
  residual[counter++] = standing * ((goal_distance_factor * mju_dot(forward, forward_target, 2) - 1) + (1.0-goal_distance_factor) * mju_dot(forward, goal_forward, 2) - 1);

  // walk forward
  residual[counter++] = standing * (mju_dot(com_vel, forward_target, 2) - parameters_[1]) * goal_distance_factor;
  // ----- move feet ----- //
  double *foot_right_vel = SensorByName(model, data, "foot_right_velocity");
  double *foot_left_vel = SensorByName(model, data, "foot_left_velocity");
  double move_feet[2];
  mju_copy(move_feet, com_vel, 2);
  mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  mju_copy(&residual[counter], move_feet, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);
  counter += 2;

  // ----- control ----- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- feet distance ----- //
  double feet_axis[2];
  mju_copy(feet_axis, foot_right, 2);
  mju_addToScl(feet_axis, foot_left, -1, 2);
  double feet_distance = mju_norm(feet_axis, 2);
  residual[counter] = feet_distance - parameters_[2];
  counter += 1;

  // ----- leg cross ----- //
  //double *left_foot_left_axis = SensorByName(model, data, "foot_left_left");
  double *right_hip_roll = SensorByName(model, data, "right_hip_roll");
  double *left_hip_roll = SensorByName(model, data, "left_hip_roll");

  // mju_sub3(vec, foot_right, foot_left);
  // residual[counter++] = mju_dot3(vec, left_foot_left_axis) + 0.3;
  residual[counter++] = *right_hip_roll - 0.15;
  residual[counter++] = -(*left_hip_roll) - 0.15;

  // ----- slippage ----- //
  double *foot_right_ang_velocity =
      SensorByName(model, data, "foot_right_ang_velocity");
  double *foot_left_ang_velocity =
      SensorByName(model, data, "foot_left_ang_velocity");
  double *right_foot_xbody = SensorByName(model, data, "foot_right_xbody");
  double *left_foot_xbody = SensorByName(model, data, "foot_left_xbody");

  residual[counter++] = (tanh(-(right_foot_xbody[2] - 0.0645) / 0.001) + 1) *
                        0.5 * foot_right_ang_velocity[2];
  residual[counter++] = (tanh(-(left_foot_xbody[2] - 0.0645) / 0.001) + 1) *
                        0.5 * foot_left_ang_velocity[2];

  // sensor dim sanity check
  // TODO: use this pattern everywhere and make this a utility function
  int user_sensor_dim = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      user_sensor_dim += model->sensor_dim[i];
    }
  }
  if (user_sensor_dim != counter) {
    mju_error_i("mismatch between total user-sensor dimension"
                "and actual length of residual %d",
                counter);
  }
}

} // namespace mjpc::h1
