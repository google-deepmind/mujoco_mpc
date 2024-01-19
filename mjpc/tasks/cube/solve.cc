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

#include "mjpc/tasks/cube/solve.h"

#include <random>
#include <string>

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string CubeSolve::XmlPath() const { return GetModelPath("cube/task.xml"); }
std::string CubeSolve::Name() const { return "Cube Solving"; }

// ---------- Residuals for cube solving manipulation task ----
//   Number of residuals:
// ------------------------------------------------------------
void CubeSolve::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                     double* residual) const {
  // initialize counter
  int counter = 0;

  // lock current mode
  int mode = current_mode_;

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

  // ---------- Residual (3) ----------
  if (mode == kModeManual || mode == kModeSolve) {
    residual[counter + 0] = data->qpos[11] - parameters_[0];  // red
    residual[counter + 1] = data->qpos[12] - parameters_[1];  // orange
    residual[counter + 2] = data->qpos[13] - parameters_[2];  // blue
    residual[counter + 3] = data->qpos[14] - parameters_[3];  // green
    residual[counter + 4] = data->qpos[15] - parameters_[4];  // white
    residual[counter + 5] = data->qpos[16] - parameters_[5];  // yellow
  } else {
    mju_zero(residual + counter, 6);
  }
  counter += 6;

  // ---------- Residual (4) ----------
  mju_sub(residual + counter, data->qpos + 97, model->key_qpos + 97, 24);
  counter += 24;

  // ---------- Residual (5) ----------
  mju_copy(residual + counter, data->qvel + 97, 24);
  counter += 24;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

// ----- Transition for cube solving manipulation task -----
//   If cube is within tolerance or floor ->
//   reset cube into hand.
// ---------------------------------------------------------
void CubeSolve::TransitionLocked(mjModel* model, mjData* data) {
  if (transition_model_) {
    if (mode == kModeWait) {
      // wait
    } else if (mode == kModeScramble) {  // scramble
      // reset
      mju_copy(data->qpos, model->qpos0, model->nq);
      mj_resetData(transition_model_, transition_data_);

      // resize
      face_.resize(num_scramble_);
      direction_.resize(num_scramble_);
      goal_cache_.resize(6 * num_scramble_);

      // set transition model
      for (int i = 0; i < num_scramble_; i++) {
        // copy goal face orientations
        mju_copy(goal_cache_.data() + i * 6, transition_data_->qpos, 6);

        // random face + direction
        std::random_device rd;  // Only used once to initialise (seed) engine
        std::mt19937 rng(
            rd());  // Random-number engine used (Mersenne-Twister in this case)

        std::uniform_int_distribution<int> uni_face(0,
                                                    5);  // Guaranteed unbiased
        face_[i] = uni_face(rng);

        std::uniform_int_distribution<int> uni_direction(
            0, 1);  // Guaranteed unbiased
        direction_[i] = uni_direction(rng);
        if (direction_[i] == 0) {
          direction_[i] = -1;
        }

        // set
        for (int t = 0; t < 2000; t++) {
          transition_data_->ctrl[face_[i]] = direction_[i] * 1.57 * t / 2000;
          mj_step(transition_model_, transition_data_);
          mju_copy(data->qpos + 11, transition_data_->qpos, 86);
        }
      }

      // set face goal index
      goal_index_ = num_scramble_ - 1;

      // set to wait
      mode = 0;
    }

    if (mode == kModeSolve) {  // solve
      // set goal
      mju_copy(parameters.data(), goal_cache_.data() + 6 * goal_index_, 6);

      // check error
      double error[6];
      mju_sub(error, data->qpos + 11, parameters.data(), 6);

      if (mju_norm(error, 6) < 0.1) {
        if (goal_index_ == 0) {
          // return to wait
          printf("solved!");
          mode = 0;
        } else {
          goal_index_--;
        }
      }
    }
  }

  // check for mode change
  if (residual_.current_mode_ != mode) {
    // update mode for residual
    residual_.current_mode_ = mode;
  }
}

}  // namespace mjpc
