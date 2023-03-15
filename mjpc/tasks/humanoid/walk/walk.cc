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

#include "mjpc/tasks/humanoid/walk/walk.h"

#include <mujoco/mujoco.h>

#include <iostream>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string humanoid::Walk::XmlPath() const {
  return GetModelPath("humanoid/walk/task.xml");
}
std::string humanoid::Walk::Name() const { return "Humanoid Walk"; }

// ------------------ Residuals for humanoid gait task ------------
  //   Number of residuals: 11
  //     Residual (0): torso height
  //     Residual (1): actuation
  //     Residual (2): balance
  //     Residual (3): upright
  //     Residual (4): posture
  //     Residual (5): goal-position error
  //     Residual (6): goal-direction error
  //     Residual (7): feet velocity
  //     Residual (8): body velocity
  //     Residual (9): gait feet height
  //     Residual (10): center-of-mass xy velocity
  //   Number of parameters: 5
  //     Parameter (0): torso height 
  //     Parameter (1): walking speed
  //     Parameter (2): walking cadence
  //     Parameter (3): walking gait feet amplitude
  //     Parameter (4): walking gait cadence
  // ----------------------------------------------------------------
void humanoid::Walk::Residual(const mjModel* model, const mjData* data,
                              double* residual) const {
  int counter = 0;

  // position error
  double* torso_position = SensorByName(model, data, "torso_position");
  double* goal = SensorByName(model, data, "goal_position");
  double goal_position_error[2];

  // goal position error
  mju_sub(goal_position_error, goal, torso_position, 2);

  // set speed terms
  double vel_scaling = 0.1;
  double speed = parameters[speed_param_id_];

  // ----- height ----- //
  double torso_height = SensorByName(model, data, "torso_position")[2];
  double* foot_right = SensorByName(model, data, "foot_right");
  double* foot_left = SensorByName(model, data, "foot_left");

  double foot_height_avg = 0.5 * (foot_right[2] + foot_left[2]);

  residual[counter++] =
    (torso_height - foot_height_avg) - parameters[torso_height_param_id_];

  // ----- actuation ----- //
  mju_copy(&residual[counter], data->actuator_force, model->nu);
  counter += model->nu;

  // ----- balance ----- //
  // capture point
  double* subcom = SensorByName(model, data, "torso_subcom");
  double* subcomvel = SensorByName(model, data, "torso_subcomvel");

  double capture_point[2];
  mju_addScl(capture_point, subcom, subcomvel, vel_scaling, 2);

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

  counter += 1;

  // ----- upright ----- //
  double* torso_up = SensorByName(model, data, "torso_up");
  double* pelvis_up = SensorByName(model, data, "pelvis_up");
  double* foot_right_up = SensorByName(model, data, "foot_right_up");
  double* foot_left_up = SensorByName(model, data, "foot_left_up");

    // torso
    residual[counter++] = torso_up[2] - 1.0;

    // pelvis
    residual[counter++] = 0.3 * (pelvis_up[2] - 1.0);

    double z_ref[3] = {0.0, 0.0, 1.0};

    // right foot
    mju_sub3(&residual[counter], foot_right_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;

    mju_sub3(&residual[counter], foot_left_up, z_ref);
    mju_scl3(&residual[counter], &residual[counter], 0.1 * standing);
    counter += 3;


  // ----- posture ----- //
  mju_sub(&residual[counter], data->qpos + 7,
          model->key_qpos + qpos_reference_id_ * model->nq + 7, model->nq - 7);
  counter += model->nq - 7;

  // ----- position error ----- //
  mju_copy(&residual[counter], goal_position_error, 2);
  counter += 2;

  // ----- orientation error ----- //
  // direction to goal
  double goal_direction[2];
  mju_copy(goal_direction, goal_position_error, 2);
  mju_normalize(goal_direction, 2);

  // torso direction
  double* torso_xaxis = SensorByName(model, data, "torso_xaxis");

  mju_sub(&residual[counter], goal_direction, torso_xaxis, 2);
  counter += 2;

  // ----- walk ----- //
  double* torso_forward = SensorByName(model, data, "torso_forward");
  double* pelvis_forward = SensorByName(model, data, "pelvis_forward");
  double* foot_right_forward = SensorByName(model, data, "foot_right_forward");
  double* foot_left_forward = SensorByName(model, data, "foot_left_forward");

  double forward[2];
  mju_copy(forward, torso_forward, 2);
  mju_addTo(forward, pelvis_forward, 2);
  mju_addTo(forward, foot_right_forward, 2);
  mju_addTo(forward, foot_left_forward, 2);
  mju_normalize(forward, 2);

  // com vel
  double* waist_lower_subcomvel =
      SensorByName(model, data, "waist_lower_subcomvel");
  double* torso_velocity = SensorByName(model, data, "torso_velocity");
  double com_vel[2];
  mju_add(com_vel, waist_lower_subcomvel, torso_velocity, 2);
  mju_scl(com_vel, com_vel, 0.5, 2);

  // walk forward
  residual[counter++] = standing * (mju_dot(com_vel, forward, 2) - speed);

  // ----- move feet ----- //
  double* foot_right_vel = SensorByName(model, data, "foot_right_velocity");
  double* foot_left_vel = SensorByName(model, data, "foot_left_velocity");
  double move_feet[2];
  mju_copy(move_feet, com_vel, 2);
  mju_addToScl(move_feet, foot_right_vel, -0.5, 2);
  mju_addToScl(move_feet, foot_left_vel, -0.5, 2);

  mju_copy(&residual[counter], move_feet, 2);
  mju_scl(&residual[counter], &residual[counter], standing, 2);

  counter += 2;

  // ----- gait ----- //
  double step[2];
  FootStep(step, GetPhase(data->time));

  double foot_pos[2][3];
  mju_copy(foot_pos[0], foot_left, 3);
  mju_copy(foot_pos[1], foot_right, 3);

  for (int i = 0; i < 2; i++) {
    double query[3] = {foot_pos[i][0], foot_pos[i][1], foot_pos[i][2]};
    double ground_height = Ground(model, data, query);
    double height_target = ground_height + 0.025 + step[i];
    double height_difference = foot_pos[i][2] - height_target;
    residual[counter++] = step[i] ? height_difference : 0;
  }

  // ----- COM xy velocity should be 0 ----- //
  mju_copy(&residual[counter], subcomvel, 2);
  counter += 2;

  // sensor dim sanity check
  CheckSensorDim(model, counter);
}

// transition
void humanoid::Walk::Transition(const mjModel* model, mjData* data) {
  // set weights and residual parameters
  if (stage != current_mode_) {
    mju_copy(weight.data(), kModeWeight[stage], weight.size());
    mju_copy(parameters.data(), kModeParameter[stage], 5);
  }

  // ---------- handle mjData reset ----------
  if (data->time < last_transition_time_ || last_transition_time_ == -1) {
    last_transition_time_ = phase_start_time_ = phase_start_ = data->time;
  }

  // ---------- prevent forbidden stage transitions ----------
  // switching stage, not from humanoid
  if (stage != current_mode_ && current_mode_ != kModeStand) {
    // switch into stateful stage only allowed from Quadruped
    if (stage == kModeWalk) {
      stage = kModeStand;
    }
  }

  // ---------- handle phase velocity change ----------
  double phase_velocity = 2 * mjPI * parameters[cadence_param_id_];
  if (phase_velocity != phase_velocity_) {
    phase_start_ = GetPhase(data->time);
    phase_start_time_ = data->time;
    phase_velocity_ = phase_velocity;
  }

  // ---------- Walk ----------
  double* goal_pos = data->mocap_pos + 3 * goal_mocap_id_;
  if (stage == kModeWalk) {
    double angvel = parameters[ParameterIndex(model, "Walk Turn")];
    double speed = parameters[ParameterIndex(model, "Walk Speed")];

    // current torso direction
    double* torso_xmat = data->xmat + 9 * torso_body_id_;
    double forward[2] = {torso_xmat[0], torso_xmat[3]};
    mju_normalize(forward, 2);
    double leftward[2] = {-forward[1], forward[0]};

    // switching into Walk or parameters changed, reset task state
    if (stage != current_mode_ || angvel_ != angvel || speed_ != speed) {
      // save time
      mode_start_time_ = data->time;

      // save current speed and angvel
      speed_ = speed;
      angvel_ = angvel;

      // compute and save rotation axis / walk origin
      double axis[2] = {data->xpos[3*torso_body_id_],
                        data->xpos[3*torso_body_id_+1]};
      if (mju_abs(angvel) > kMinAngvel) {
        // don't allow turning with very small angvel
        double d = speed / angvel;
        axis[0] += d * leftward[0];
        axis[1] += d * leftward[1];
      }
      position_[0] = axis[0];
      position_[1] = axis[1];

      // save vector from axis to initial goal position
      heading_[0] = goal_pos[0] - axis[0];
      heading_[1] = goal_pos[1] - axis[1];
    }

    // move goal
    double time = data->time - mode_start_time_;
    WalkPosition(goal_pos, time);
  }

  // save stage
  current_mode_ = static_cast<HumanoidMode>(stage);
  last_transition_time_ = data->time;
}

// reset humanoid task
void humanoid::Walk::Reset(const mjModel* model) {
  // call method from base class
  Task::Reset(model);

  // ----------  task identifiers  ----------
  torso_height_param_id_ = ParameterIndex(model, "Torso");
  speed_param_id_ = ParameterIndex(model, "Speed");
  cadence_param_id_ = ParameterIndex(model, "Cadence");
  amplitude_param_id_ = ParameterIndex(model, "Amplitude");
  duty_param_id_ = ParameterIndex(model, "DutyRatio");
  balance_cost_id_ = CostTermByName(model, "Balance");
  upright_cost_id_ = CostTermByName(model, "Upright");
  height_cost_id_ = CostTermByName(model, "Height");
  qpos_reference_id_ = 0;

  // ----------  model identifiers  ----------
  torso_body_id_ = mj_name2id(model, mjOBJ_XBODY, "torso");
  if (torso_body_id_ < 0) mju_error("body 'torso' not found");

  head_site_id_ = mj_name2id(model, mjOBJ_XBODY, "head");
  if (head_site_id_ < 0) mju_error("body 'head' not found");

  int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (goal_id < 0) mju_error("body 'goal' not found");

  goal_mocap_id_ = model->body_mocapid[goal_id];
  if (goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");
}

// colors of visualisation elements drawn in ModifyScene()
constexpr float kCapRgba[4] = {1.0, 0.0, 1.0, 1.0};  // capture point
constexpr float kHullRgba[4] = {1.0, 0.0, 0.0, 1};   // convex hull
constexpr float kPcpRgba[4] = {1.0, 0.0, 0.0, 1.0};  // projected capture point

// draw task-related geometry in the scene
void humanoid::Walk::ModifyScene(const mjModel* model, const mjData* data,
                                 mjvScene* scene) const {
  // feet site positions (xy plane)
  double foot_pos[4][3];
  mju_copy(foot_pos[0], SensorByName(model, data, "sp0"), 3);
  mju_copy(foot_pos[1], SensorByName(model, data, "sp1"), 3);
  mju_copy(foot_pos[2], SensorByName(model, data, "sp2"), 3);
  mju_copy(foot_pos[3], SensorByName(model, data, "sp3"), 3);
  foot_pos[0][2] = 0.0;
  foot_pos[1][2] = 0.0;
  foot_pos[2][2] = 0.0;
  foot_pos[3][2] = 0.0;

  // support polygon
  double polygon[8];
  for (int i = 0; i < 4; i++) {
    polygon[2 * i] = foot_pos[i][0];
    polygon[2 * i + 1] = foot_pos[i][1];
  }
  int hull[4];
  int num_hull = Hull2D(hull, 4, polygon);

  // draw connectors
  for (int i = 0; i < num_hull; i++) {
    int j = (i + 1) % num_hull;
    AddConnector(scene, mjGEOM_CAPSULE, 0.015, foot_pos[hull[i]],
                 foot_pos[hull[j]], kHullRgba);
  }

  // capture point
  double fall_time = mju_sqrt(2.0 * parameters[torso_height_param_id_] /
                              mju_norm(model->opt.gravity, 3));
  double capture[3];
  double* compos = SensorByName(model, data, "torso_subcom");
  double* comvel = SensorByName(model, data, "torso_subcomvel");
  mju_addScl3(capture, compos, comvel, fall_time);

  // ground under CoM
  double com_ground = Ground(model, data, compos);

  // capture point
  double foot_size[3] = {kFootRadius, 0, 0};

  capture[2] = com_ground;

  AddGeom(scene, mjGEOM_SPHERE, foot_size, capture, /*mat=*/nullptr, kCapRgba);

  // capture point, projected onto hull
  double pcp2[2];
  NearestInHull(pcp2, capture, polygon, hull, num_hull);
  double pcp[3] = {pcp2[0], pcp2[1], com_ground};
  AddGeom(scene, mjGEOM_SPHERE, foot_size, pcp, /*mat=*/nullptr, kPcpRgba);
}

// return phase as a function of time
double humanoid::Walk::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

// horizontal Walk trajectory
void humanoid::Walk::WalkPosition(double pos[2], double time) const {
  if (mju_abs(angvel_) < kMinAngvel) {
    // no rotation, go in straight line
    double forward[2] = {heading_[0], heading_[1]};
    mju_normalize(forward, 2);
    pos[0] = position_[0] + heading_[0] + time*speed_*forward[0];
    pos[1] = position_[1] + heading_[1] + time*speed_*forward[1];
  } else {
    // walk on a circle
    double angle = time * angvel_;
    double mat[4] = {mju_cos(angle), -mju_sin(angle),
                     mju_sin(angle),  mju_cos(angle)};
    mju_mulMatVec(pos, mat, heading_, 2, 2);
    pos[0] += position_[0];
    pos[1] += position_[1];
  }
}

// return normalized target step height
double humanoid::Walk::StepHeight(double time, double footphase,
                                  double duty_ratio) const {
  double angle = std::fmod(time + mjPI - footphase, 2 * mjPI) - mjPI;
  double value = 0;
  if (duty_ratio < 1) {
    angle *= 0.5 / (1 - duty_ratio);
    value = mju_cos(mju_clip(angle, -mjPI / 2, mjPI / 2));
  }
  return mju_abs(value) < 1e-6 ? 0.0 : value;
}

// compute target step height for all feet
void humanoid::Walk::FootStep(double* step, double time) const {
  double amplitude = parameters[amplitude_param_id_];
  double duty_ratio = parameters[duty_param_id_];
  double gait_phase[2] = {0.0, 0.5};
  for (int i = 0; i < 2; i++) {
    double footphase = 2 * mjPI * gait_phase[i];
    step[i] = amplitude * StepHeight(time, footphase, duty_ratio);
  }
}

}  // namespace mjpc
