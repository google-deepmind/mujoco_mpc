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

#include "mjpc/tasks/leap/leap.h"

#include <mujoco/mujoco.h>

#include <string>

#include "mjpc/utilities.h"

namespace mjpc {
std::string Leap::XmlPath() const { return GetModelPath("leap/task.xml"); }
std::string Leap::Name() const { return "Leap"; }

// ------- Residuals for cube manipulation task ------
//     Cube position: (3)
//     Cube orientation: (3)
//     Cube linear velocity: (3)
//     Control: (16), there are 16 servos
//     Nominal pose: (16)
//     Joint velocity: (16)
// ------------------------------------------
void Leap::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                double *residual) const {
  int counter = 0;

  // ---------- Cube position ----------
  // we use the kRectifyLoss on the distance of the cube to a rectangular tube of the following dimensions:
  // * x in [0.08, 0.14]
  // * y in [-0.02, 0.025]
  // the loss has the form y = p * log(1 + exp(var / p))
  // we let var = 250 * dist(cube_center, tube),
  // where the distance function reports a set distance and 250 is a tuned slope controlling the scale of the loss.
  // for p = 0.05, this gives roughly a loss of 1 with a 1cm violation.
  double *cube_position = SensorByName(model, data, "cube_position");

  double x = cube_position[0];
  double y = cube_position[1];
  double z = cube_position[2];

  double x_min = 0.08;
  double x_max = 0.14;
  double y_min = -0.02;
  double y_max = 0.02;

  double x_closest = mju_max(x_min, mju_min(x, x_max));
  double y_closest = mju_max(y_min, mju_min(y, y_max));

  // double dist =
  //     std::sqrt(std::pow(x_closest - cube_position[0], 2) + std::pow(y_closest - cube_position[1], 2));

  // if (z <= 0.15 && x <= x_min && x >= x_max && y <= y_min && y >= y_max) {
  //   dist += 10.0;  // dropping is bad
  // }
  double z_closest;
  if (x < x_min || x > x_max || y < y_min || y > y_max) {
      double theta = 0.349066;  // 20 degree palm tilt
      double z_min = x * std::tan(theta) - 0.035 / std::cos(theta);  // height of center of cube if flat
      double z_max = z_min + 0.035;  // allow the cube to come up a bit
      z_closest = mju_max(z_min, mju_min(z, z_max));
  } else {
      double z_min = 0.015;
      z_closest = mju_max(z_min, z);
  }

  double dist = std::sqrt(
      std::pow(x_closest - cube_position[0], 2) +
      std::pow(y_closest - cube_position[1], 2) +
      std::pow(z_closest - cube_position[2], 2));

  residual[counter] = 250.0 * dist;  // new loss takes a scalar
  counter += 1;

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

  // ---------- Cube angular velocity ----------
  double *cube_angular_velocity =
      SensorByName(model, data, "cube_angular_velocity");

  mju_copy(residual + counter, cube_angular_velocity, 3);
  counter += 3;

  // ---------- Control ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // ---------- Nominal Pose ----------
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, 16);
  counter += 16;

  // ---------- Joint Velocity ----------
  mju_copy(residual + counter, data->qvel + 6, 16);
  counter += 16;

  // Sanity check
  CheckSensorDim(model, counter);
}

void Leap::TransitionLocked(mjModel *model, mjData *data) {
  // Compute the angle between the cube and the goal orientation
  double *cube_orientation = SensorByName(model, data, "cube_orientation");
  double *goal_cube_orientation =
      SensorByName(model, data, "cube_goal_orientation");

  // if goal cube orientation is all 0s, set it to {1.0, 0.0, 0.0, 0.0}
  // do this without using a norm utility
  if (
    goal_cube_orientation[0] == 0.0 &&
    goal_cube_orientation[1] == 0.0 &&
    goal_cube_orientation[2] == 0.0 &&
    goal_cube_orientation[3] == 0.0
  ) {
    goal_cube_orientation[0] = 1.0;
    goal_cube_orientation[1] = 0.0;
    goal_cube_orientation[2] = 0.0;
    goal_cube_orientation[3] = 0.0;
  }

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
  double angle = 2.0 * std::acos(q_diff[0]) * 180.0 / M_PI;  // in degrees

  // Check if the goal orientation is axis-aligned
  int axis_aligned_goal = GetNumberOrDefault(1, model, "axis_aligned_goal");
  double angle_thresh = axis_aligned_goal ? 11.4592 : 22.9183;  // 0.2 and 0.4 radians, (0.4 same as dextreme)

  // Decide whether to change the goal orientation
  bool change_goal = false;
  if (angle < angle_thresh) {
    change_goal = true;
    ++rotation_count_;
    if (rotation_count_ > best_rotation_count_) {
      best_rotation_count_ = rotation_count_;
    }
  }

  // Figure out whether we dropped the cube
  int cube = mj_name2id(model, mjOBJ_GEOM, "cube");
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");

  bool on_floor = false;
  for (int i = 0; i < data->ncon; i++) {
    mjContact *g = data->contact + i;
    if ((g->geom1 == cube && g->geom2 == floor) ||
        (g->geom2 == cube && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  // Reset the cube to be on the hand if needed
  if (on_floor) {
    int cube_body = mj_name2id(model, mjOBJ_BODY, "cube");
    if (cube_body != -1) {
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[cube_body]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[cube_body]];
      mju_copy(data->qpos + jnt_qposadr, model->key_qpos + jnt_qposadr, 7);
      mju_zero(data->qvel + jnt_veladr, 6);
    }
  }

  // Reset the rotation count if we dropped the cube or took too long
  time_since_last_reset_ =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - time_of_last_reset_)
          .count();
  time_since_last_rotation_ =
      std::chrono::duration_cast<std::chrono::duration<double>>(
          std::chrono::steady_clock::now() - time_of_last_rotation_)
          .count();

  // [DEBUG] for now, disable the timeout
  // if (on_floor || time_since_last_rotation_ > 60.0) {  // 60 second timeout
  //   time_of_last_reset_ = std::chrono::steady_clock::now();
  //   rotation_count_ = 0;
  //   change_goal = true;
  // }

  // Change the goal orientation if needed
  if (change_goal) {
    time_of_last_rotation_ = std::chrono::steady_clock::now();

    // [option 1] uniformly randomly sample one of 24 possible cube orientations
    if (axis_aligned_goal) {
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

      // set the new goal orientation
      int goal = mj_name2id(model, mjOBJ_GEOM, "goal");
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[goal]];
      mju_copy(data->mocap_quat + jnt_qposadr, q_goal.data(), 4);

    // [option 2] uniformly randomly sample a quaternion
    // see: https://stackoverflow.com/a/44031492
    } else {
      while (true) {
        const double a = absl::Uniform<double>(gen_, 0.0, 1.0);
        const double b = absl::Uniform<double>(gen_, 0.0, 1.0);
        const double c = absl::Uniform<double>(gen_, 0.0, 1.0);
        const double s1 = std::sqrt(1.0 - a);
        const double s2 = std::sqrt(a);
        const double sb = std::sin(2.0 * mjPI * b);
        const double cb = std::cos(2.0 * mjPI * b);
        const double sc = std::sin(2.0 * mjPI * c);
        const double cc = std::cos(2.0 * mjPI * c);
        std::vector<double> q_goal = {s1 * sb, s1 * cb, s2 * sc, s2 * cc};

        // check the new goal is far enough away from the current orientation
        // only consider rots >= 90 degs
        std::vector<double> q_diff = {0.0, 0.0, 0.0, 0.0};
        mju_mulQuat(q_diff.data(), q_goal.data(), q_gco_conj.data());
        mju_normalize4(q_diff.data());
        if (q_diff[0] < 0.0) {
          q_diff[0] *= -1.0;
          q_diff[1] *= -1.0;
          q_diff[2] *= -1.0;
          q_diff[3] *= -1.0;
        }
        double angle = 2.0 * std::acos(q_diff[0]) * 180.0 / M_PI;  // in degrees

        // Set the new goal orientation
        if (angle >= 90.0) {
          int goal = mj_name2id(model, mjOBJ_GEOM, "goal");
          int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[goal]];
          mju_copy(data->mocap_quat + jnt_qposadr, q_goal.data(), 4);
          break;
        }
      }
    }
  }

  if (on_floor || change_goal) {
    // Reset stored data in the simulation
    mutex_.unlock();
    mj_forward(model, data);
    mutex_.lock();
  }

  // Update rotation counters in the GUI
  parameters[0] = rotation_count_;
  parameters[1] = best_rotation_count_;
  parameters[2] = time_since_last_rotation_;
  parameters[3] =
      time_since_last_reset_ / std::max(double(rotation_count_), 1.0);
}

}  // namespace mjpc
