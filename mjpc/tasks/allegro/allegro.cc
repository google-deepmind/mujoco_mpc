// Copyright 2024 DeepMind Technologies Limited
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

#include "mjpc/tasks/allegro/allegro.h"

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <chrono>
#include <cmath>
#include <string>

#include "mjpc/utilities.h"

namespace mjpc {
std::string Allegro::XmlPath() const {
  return GetModelPath("allegro/task.xml");
}
std::string Allegro::Name() const { return "Allegro"; }

// ------- Residuals for cube manipulation task ------
//     Cube position: (3)
//     Cube orientation: (3)
//     Cube linear velocity: (3)
//     Actuation: (16) - desired finger positions relative to reasonable nominal
// ------------------------------------------
void Allegro::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                   double *residual) const {
  int counter = 0;

  // ---------- Cube position ----------
  double *cube_position = SensorByName(model, data, "cube_position");
  double *cube_goal_position = SensorByName(model, data, "cube_goal_position");

  // difference between the cube position and goal position
  mju_sub3(residual + counter, cube_position, cube_goal_position);

  // penalty if the cube's x dimension is outside the hand/on edges
  if (cube_position[0] < -0.1 + 0.140696023 ||
      cube_position[0] > 0.08 + 0.140696023) {
    residual[counter] *= 10.0;
  }

  // penalty if the cube's y dimension is near edges
  if (cube_position[1] < -0.05 - 0.005106107 ||
      cube_position[1] > 0.03 - 0.005106107) {
    residual[counter + 1] *= 10.0;
  }
  if (cube_position[2] < -0.03 - 0.048353794) {
    residual[counter + 2] *= 10.0;
  }

  counter += 3;

  // ---------- Cube orientation ----------
  double *cube_orientation = SensorByName(model, data, "cube_orientation");
  double *goal_cube_orientation =
      SensorByName(model, data, "cube_goal_orientation");
  mju_normalize4(goal_cube_orientation);

  mju_subQuat(residual + counter, goal_cube_orientation, cube_orientation);
  counter += 3;

  // ---------- Cube linear velocity ----------
  double *cube_linear_velocity =
      SensorByName(model, data, "cube_linear_velocity");

  mju_copy(residual + counter, cube_linear_velocity, 3);
  counter += 3;

  // ---------- Actuation ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // Sanity check
  CheckSensorDim(model, counter);
}

void Allegro::TransitionLocked(mjModel *model, mjData *data) {
  // Check for contact between the cube and the floor
  int cube = mj_name2id(model, mjOBJ_GEOM, "cube");
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");
  bool on_floor = false;
  bool new_goal = false;
  for (int i = 0; i < data->ncon; i++) {
    mjContact *g = data->contact + i;
    if ((g->geom1 == cube && g->geom2 == floor) ||
        (g->geom2 == cube && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  // If the cube is on the floor and not moving, reset it
  double *cube_lin_vel = SensorByName(model, data, "cube_linear_velocity");

  // If timeout has been reached, reset
  auto duration = std::chrono::duration<double>(
                      std::chrono::steady_clock::now() - time_reset)
                      .count();

  if ((on_floor && mju_norm3(cube_lin_vel) < 0.001) || duration > timeout_) {
    // reset the timeout if timed out
    if (duration > timeout_) {
      time_reset = std::chrono::steady_clock::now();
    }

    int cube_body = mj_name2id(model, mjOBJ_BODY, "cube");
    if (cube_body != -1) {
      // reset cube
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[cube_body]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[cube_body]];
      mju_copy(data->qpos + jnt_qposadr, model->key_qpos + jnt_qposadr, 7);
      mju_zero(data->qvel + jnt_veladr, 6);

      // reset palm
      int palm_qposadr = 11;  // goal quat then cube pose come first
      int palm_veladr = 9;
      mju_copy(data->qpos + palm_qposadr, model->qpos0 + palm_qposadr, 16);
      mju_zero(data->qvel + palm_veladr, 16);

      // reset counter
      if (rotation_counter > num_best_rots) {
        num_best_rots = rotation_counter;
      }
      prev_best_rots = rotation_counter;
      rotation_counter = 0;
    }
  }

  // If the orientation of the cube is close to the goal, change the goal
  double *cube_orientation = SensorByName(model, data, "cube_orientation");
  double *goal_cube_orientation =
      SensorByName(model, data, "cube_goal_orientation");
  std::vector<double> q_diff = {0.0, 0.0, 0.0, 0.0};
  std::vector<double> q_gco_conj = {0.0, 0.0, 0.0, 0.0};
  mju_negQuat(q_gco_conj.data(), goal_cube_orientation);
  mju_mulQuat(q_diff.data(), cube_orientation, q_gco_conj.data());
  mju_normalize4(q_diff.data());
  if (q_diff[0] < 0.0) {
    q_diff[0] *= -1.0;
    q_diff[1] *= -1.0;
    q_diff[2] *= -1.0;
    q_diff[3] *= -1.0;
  }

  // if within 15 degrees of goal orientation, change goal
  double angle = 2.0 * std::acos(q_diff[0]);
  if (angle <= 0.261799 && duration > 0.05) {
    // don't allow resetting super fast to avoid double counting
    time_reset = std::chrono::steady_clock::now();

    // for the average time per rotation, don't count first rot
    if (first_rot) {
      first_rot = false;
      time_start = std::chrono::steady_clock::now();
    } else {
      total_rots += 1;
      time_per_rot = std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - time_start)
                         .count() /
                     total_rots;
    }

    // advance the rotation counter
    rotation_counter += 1;

    // sampling token
    absl::BitGen gen_;

    // [option] sample new goal quaternion uniformly
    // https://stackoverflow.com/a/44031492
    // double a = absl::Uniform<double>(gen_, 0.0, 1.0);
    // double b = absl::Uniform<double>(gen_, 0.0, 1.0);
    // double c = absl::Uniform<double>(gen_, 0.0, 1.0);
    // double s1 = std::sqrt(1.0 - a);
    // double s2 = std::sqrt(a);
    // double sb = std::sin(2.0 * mjPI * b);
    // double cb = std::cos(2.0 * mjPI * b);
    // double sc = std::sin(2.0 * mjPI * c);
    // double cc = std::cos(2.0 * mjPI * c);
    // std::vector<double> q_goal = {s1 * sb, s1 * cb, s2 * sc, s2 * cc};

    // [option] uniformly randomly sample one of 24 possible cube orientations
    std::vector<double> q0 = {0.0, 1.0, 0.0, 0.7};      // wrist tilt
    std::vector<double> q1 = {0.0, 0.0, 0.0, 0.0};      // first rotation
    std::vector<double> q2 = {0.0, 0.0, 0.0, 0.0};      // second rotation
    std::vector<double> q_goal = {0.0, 0.0, 0.0, 0.0};  // goal rotation

    // ensure that the newly sampled rotation differs from the old one
    int rand1 = rand1_;
    int rand2 = rand2_;
    while (rand1 == rand1_ && rand2 == rand2_) {
      rand1 = absl::Uniform<int>(gen_, 0, 6);
      rand2 = absl::Uniform<int>(gen_, 0, 4);
    }
    rand1_ = rand1;  // reset the old rotation
    rand2_ = rand2;

    // choose which face faces +z
    if (rand1 == 0) {
      // do nothing
      q1 = {1.0, 0.0, 0.0, 0.0};
    } else if (rand1 == 1) {
      // rotate about x axis by 90 degrees
      q1 = {0.7071067811865476, 0.7071067811865476, 0.0, 0.0};
    } else if (rand1 == 2) {
      // rotate about x axis by 180 degrees
      q1 = {0.0, 1.0, 0.0, 0.0};
    } else if (rand1 == 3) {
      // rotate about x axis by 270 degrees
      q1 = {-0.7071067811865476, 0.7071067811865476, 0.0, 0.0};
    } else if (rand1 == 4) {
      // rotate about y axis by 90 degrees
      q1 = {0.7071067811865476, 0.0, 0.7071067811865476, 0.0};
    } else if (rand1 == 5) {
      // rotate about y axis by 270 degrees
      q1 = {0.7071067811865476, 0.0, -0.7071067811865476, 0.0};
    }

    // choose rotation about +z
    if (rand2 == 0) {
      // do nothing
      q2 = {1.0, 0.0, 0.0, 0.0};
    } else if (rand2 == 1) {
      // rotate about z axis by 90 degrees
      q2 = {0.7071067811865476, 0.0, 0.0, 0.7071067811865476};
    } else if (rand2 == 2) {
      // rotate about z axis by 180 degrees
      q2 = {0.0, 0.0, 0.0, 1.0};
    } else if (rand2 == 3) {
      // rotate about z axis by 270 degrees
      q2 = {-0.7071067811865476, 0.0, 0.0, 0.7071067811865476};
    }

    // combine the two quaternions
    mju_mulQuat(q_goal.data(), q0.data(), q2.data());
    mju_mulQuat(q_goal.data(), q_goal.data(), q1.data());
    mju_normalize4(q_goal.data());  // enforce unit norm

    // set new goal quaternion
    mju_copy(data->mocap_quat, q_goal.data(), 4);
  }

  if (on_floor || new_goal) {
    // Step the simulation forward
    mutex_.unlock();
    mj_forward(model, data);
    mutex_.lock();
  }

  // update mocap position
  // [DEBUG] if employing debug hacks in app.cc, this must be commented out!
  std::vector<double> pos_cube = pos_cube_;
  std::vector<double> quat_cube = quat_cube_;
  mju_copy(data->mocap_pos + 3, pos_cube.data(), 3);
  mju_copy(data->mocap_quat + 4, quat_cube.data(), 4);

  // update the rotation counter in the GUI
  parameters[5] = rotation_counter;
  parameters[6] = num_best_rots;
  parameters[7] = prev_best_rots;
  parameters[8] = total_rots;
  parameters[9] = time_per_rot;
}

void Allegro::ModifyState(const mjModel *model, State *state) {
  // sampling token
  absl::BitGen gen_;

  // std from GUI
  double std_rot =
      parameters[0];  // stdev for rotational noise in tangent space
  double std_pos = parameters[1];    // uniform stdev for position noise
  double bias_posx = parameters[2];  // bias for position noise
  double bias_posy = parameters[3];  // bias for position noise
  double bias_posz = parameters[4];  // bias for position noise

  // current state
  const std::vector<double> &s = state->state();

  // add quaternion noise
  std::vector<double> dv = {0.0, 0.0, 0.0};  // rotational velocity noise
  dv[0] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  dv[1] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  dv[2] = absl::Gaussian<double>(gen_, 0.0, std_rot);
  std::vector<double> quat_cube = {s[3], s[4], s[5], s[6]};  // quat cube state
  mju_quatIntegrate(quat_cube.data(), dv.data(), 1.0);        // update the quat
  mju_normalize4(quat_cube.data());  // normalize the quat for numerics

  // add position noise
  std::vector<double> dp = {bias_posx, bias_posy,
                            bias_posz};  // translational velocity noise
  dp[0] += absl::Gaussian<double>(gen_, 0.0, std_pos);
  dp[1] += absl::Gaussian<double>(gen_, 0.0, std_pos);
  dp[2] += absl::Gaussian<double>(gen_, 0.0, std_pos);
  std::vector<double> pos_cube = {s[0], s[1], s[2]};  // position cube state
  mju_addTo3(pos_cube.data(), dp.data());             // update the pos

  // set state
  std::vector<double> qpos(model->nq);
  mju_copy(qpos.data(), s.data(), model->nq);
  mju_copy(qpos.data() + 0, pos_cube.data(), 3);
  mju_copy(qpos.data() + 3, quat_cube.data(), 4);
  state->SetPosition(model, qpos.data());

  // update cube mocap state
  mju_copy(pos_cube_.data(), pos_cube.data(), 3);
  mju_copy(quat_cube_.data(), quat_cube.data(), 4);
}

// Change the friction coefficient of all the objects in the scene
void Allegro::DomainRandomize(std::vector<mjModel *> &randomized_models) const {
  absl::BitGen gen_;

  // Standard deviations are set by slider parameters
  double friction_std_dev = parameters[10];
  double act_gain_std_dev = parameters[11];
  double cube_pos_std_dev = parameters[12];

  // Each model has all friction coefficients boosted or shrunk, so some models
  // are more slippery and others are more grippy.
  for (int i = 0; i < randomized_models.size(); i++) {
    mjModel *model = randomized_models[i];

    const double friction_change =
        absl::Gaussian<double>(gen_, 0.0, friction_std_dev);
    for (int j = 0; j < model->ngeom; j++) {
      model->geom_friction[j] += friction_change;
      model->geom_friction[j] = std::max(model->geom_friction[j], 0.0);
    }
  }

  // Each model has different acutator gains
  for (int i = 0; i < randomized_models.size(); i++) {
    mjModel *model = randomized_models[i];

    const double act_gain_change =
        absl::Gaussian<double>(gen_, 0.0, act_gain_std_dev);
    for (int j = 0; j < model->nu; j++) {
      model->actuator_gainprm[2 * j] += act_gain_change;
      model->actuator_gainprm[2 * j] =
          std::max(model->actuator_gainprm[2 * j], 0.01);
    }
  }

  // The cube is in a different position in each model
  const int cube_body_id = mj_name2id(randomized_models[0], mjOBJ_BODY, "cube");

  for (int i = 0; i < randomized_models.size(); ++i) {
    mjModel *model = randomized_models[i];

    const double cube_dx = absl::Gaussian<double>(gen_, 0.0, cube_pos_std_dev);
    const double cube_dy = absl::Gaussian<double>(gen_, 0.0, cube_pos_std_dev);
    const double cube_dz = absl::Gaussian<double>(gen_, 0.0, cube_pos_std_dev);

    model->body_pos[3 * cube_body_id] += cube_dx;
    model->body_pos[3 * cube_body_id + 1] += cube_dy;
    model->body_pos[3 * cube_body_id + 2] += cube_dz;
  }
}

}  // namespace mjpc
