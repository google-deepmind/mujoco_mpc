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

#include "mjpc/tasks/quadruped/quadruped.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string QuadrupedHill::XmlPath() const {
  return GetModelPath("quadruped/task_hill.xml");
}
std::string QuadrupedFlat::XmlPath() const {
  return GetModelPath("quadruped/task_flat.xml");
}
std::string QuadrupedHill::Name() const { return "Quadruped Hill"; }
std::string QuadrupedFlat::Name() const { return "Quadruped Flat"; }


void QuadrupedFlat::Residual(const mjModel* model, const mjData* data,
                             double* residual) const {
  // start counter
  int counter = 0;

  // get foot positions
  double* foot_pos[kNumFoot];
  for (A1Foot foot : kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];

  double* shoulder_pos[kNumFoot];
  for (A1Foot foot : kFootAll)
    shoulder_pos[foot] = data->xpos + 3 * shoulder_body_id_[foot];

  // average foot position
  double avg_foot_pos[3];
  AverageFootPos(avg_foot_pos, foot_pos);

  double* torso_xmat = data->xmat + 9*torso_body_id_;
  double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
  double* compos = SensorByName(model, data, "torso_subtreecom");


  // ---------- Upright ----------
  if (current_mode_ != kModeFlip) {
    if (current_mode_ == kModeBiped) {
      double biped_type = parameters[biped_type_param_id_];
      int handstand = ReinterpretAsInt(biped_type) ? -1 : 1;
      residual[counter++] = torso_xmat[6] - handstand;
    } else {
      residual[counter++] = torso_xmat[8] - 1;
    }
    residual[counter++] = 0;
    residual[counter++] = 0;
  } else {
    // special handling of flip orientation
    double flip_time = data->time - mode_start_time_;
    double quat[4];
    FlipQuat(quat, flip_time);
    double* torso_xquat = data->xquat + 4*torso_body_id_;
    mju_subQuat(residual + counter, torso_xquat, quat);
    counter += 3;
  }


  // ---------- Height ----------
  // quadrupedal or bipedal height of torso over feet
  double* torso_pos = data->xipos + 3*torso_body_id_;
  bool is_biped = current_mode_ == kModeBiped;
  double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;
  if (current_mode_ == kModeScramble) {
    // disable height term in Scramble
    residual[counter++] = 0;
  } else if (current_mode_ == kModeFlip) {
    // height target for Backflip
    double flip_time = data->time - mode_start_time_;
    residual[counter++] = torso_pos[2] - FlipHeight(flip_time);
  } else {
    residual[counter++] = (torso_pos[2] - avg_foot_pos[2]) - height_goal;
  }


  // ---------- Position ----------
  double* head = data->site_xpos + 3*head_site_id_;
  double target[3];
  if (mode == kModeWalk) {
    // follow prescribed Walk trajectory
    double mode_time = data->time - mode_start_time_;
    Walk(target, mode_time);
  } else {
    // go to the goal mocap body
    target[0] = goal_pos[0];
    target[1] = goal_pos[1];
    target[2] = goal_pos[2];
  }
  residual[counter++] = head[0] - target[0];
  residual[counter++] = head[1] - target[1];
  residual[counter++] = mode == kModeScramble ? 2*(head[2] - target[2]) : 0;


  // ---------- Gait ----------
  A1Gait gait = GetGait();
  double step[kNumFoot];
  FootStep(step, GetPhase(data->time), gait);
  for (A1Foot foot : kFootAll) {
    if (is_biped) {
      // ignore "hands" in biped mode
      bool handstand = ReinterpretAsInt(parameters[biped_type_param_id_]);
      bool front_hand = !handstand && (foot == kFootFL || foot == kFootFR);
      bool back_hand = handstand && (foot == kFootHL || foot == kFootHR);
      if (front_hand || back_hand) {
        residual[counter++] = 0;
        continue;
      }
    }
    double query[3] = {foot_pos[foot][0], foot_pos[foot][1], foot_pos[foot][2]};

    if (current_mode_ == kModeScramble) {
      double torso_to_goal[3];
      double* goal = data->mocap_pos + 3*goal_mocap_id_;
      mju_sub3(torso_to_goal, goal, torso_pos);
      mju_normalize3(torso_to_goal);
      mju_sub3(torso_to_goal, goal, foot_pos[foot]);
      torso_to_goal[2] = 0;
      mju_normalize3(torso_to_goal);
      mju_addToScl3(query, torso_to_goal, 0.15);
    }

    double ground_height = Ground(model, data, query);
    double height_target = ground_height + kFootRadius + step[foot];
    double height_difference = foot_pos[foot][2] - height_target;
    if (current_mode_ == kModeScramble) {
      // in Scramble, foot higher than target is not penalized
      height_difference = mju_min(0, height_difference);
    }
    residual[counter++] = step[foot] ? height_difference : 0;
  }


  // ---------- Balance ----------
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double capture_point[3];
  double fall_time = mju_sqrt(2*height_goal / 9.81);
  mju_addScl3(capture_point, compos, comvel, fall_time);
  residual[counter++] = capture_point[0] - avg_foot_pos[0];
  residual[counter++] = capture_point[1] - avg_foot_pos[1];


  // ---------- Effort ----------
  mju_scl(residual + counter, data->actuator_force, 2e-2, model->nu);
  counter += model->nu;


  // ---------- Posture ----------
  double* home = KeyQPosByName(model, data, "home");
  mju_sub(residual + counter, data->qpos + 7, home + 7, model->nu);
  if (current_mode_ == kModeFlip) {
    double flip_time = data->time - mode_start_time_;
    if (flip_time < crouch_time_) {
      double* crouch = KeyQPosByName(model, data, "crouch");
      mju_sub(residual + counter, data->qpos + 7, crouch + 7, model->nu);
    } else if (flip_time >= crouch_time_ &&
               flip_time < jump_time_ + flight_time_) {
      // free legs during flight phase
      mju_zero(residual + counter, model->nu);
    }
  }
  for (A1Foot foot : kFootAll) {
    for (int joint = 0; joint < 3; joint++) {
      residual[counter + 3*foot + joint] *= kJointPostureGain[joint];
    }
  }
  if (current_mode_ == kModeBiped) {
    // loosen the "hands" in Biped mode
    bool handstand = ReinterpretAsInt(parameters[biped_type_param_id_]);
    if (handstand) {
      residual[counter + 4] *= 0.03;
      residual[counter + 5] *= 0.03;
      residual[counter + 10] *= 0.03;
      residual[counter + 11] *= 0.03;
    } else {
      residual[counter + 1] *= 0.03;
      residual[counter + 2] *= 0.03;
      residual[counter + 7] *= 0.03;
      residual[counter + 8] *= 0.03;
    }
  }
  counter += model->nu;


  // ---------- Yaw ----------
  double torso_heading[2] = {torso_xmat[0], torso_xmat[3]};
  if (current_mode_ == kModeBiped) {
    int handstand = ReinterpretAsInt(parameters[biped_type_param_id_]) ? 1 : -1;
    torso_heading[0] = handstand * torso_xmat[2];
    torso_heading[1] = handstand * torso_xmat[5];
  }
  mju_normalize(torso_heading, 2);
  double heading_goal = parameters[ParameterIndex(model, "Heading")];
  residual[counter++] = torso_heading[0] - mju_cos(heading_goal);
  residual[counter++] = torso_heading[1] - mju_sin(heading_goal);


  // ---------- Angular momentum ----------
  mju_copy3(residual + counter, SensorByName(model, data, "torso_angmom"));
  counter +=3;


  // sensor dim sanity check
  CheckSensorDim(model, counter);
}


//  ============  transition  ============
void QuadrupedFlat::Transition(const mjModel* model, mjData* data) {
  // ---------- handle mjData reset ----------
  if (data->time < last_transition_time_ || last_transition_time_ == -1) {
    if (mode != kModeQuadruped && mode != kModeBiped) {
      mode = kModeQuadruped;  // mode is stateful, switch to Quadruped
    }
    last_transition_time_ = phase_start_time_ = phase_start_ = data->time;
  }


  // ---------- prevent forbidden mode transitions ----------
  // switching mode, not from quadruped
  if (mode != current_mode_ && current_mode_ != kModeQuadruped) {
    // switch into stateful mode only allowed from Quadruped
    if (mode == kModeWalk || mode == kModeFlip) {
      mode = kModeQuadruped;
    }
  }


  // ---------- handle phase velocity change ----------
  double phase_velocity = 2 * mjPI * parameters[cadence_param_id_];
  if (phase_velocity != phase_velocity_) {
    phase_start_ = GetPhase(data->time);
    phase_start_time_ = data->time;
    phase_velocity_ = phase_velocity;
  }


  // ---------- automatic gait switching ----------
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  double beta = mju_exp(-(data->time-last_transition_time_) / kAutoGaitFilter);
  com_vel_[0] = beta * com_vel_[0] + (1 - beta) * comvel[0];
  com_vel_[1] = beta * com_vel_[1] + (1 - beta) * comvel[1];
  // TODO(b/268398978): remove reinterpret, int64_t business
  int auto_switch = ReinterpretAsInt(parameters[gait_switch_param_id_]);
  if (mode == kModeBiped) {
    // biped always trots
    parameters[gait_param_id_] = ReinterpretAsDouble(kGaitTrot);
  } else if (auto_switch) {
    double com_speed = mju_norm(com_vel_, 2);
    for (int64_t gait : kGaitAll) {
      // scramble requires a non-static gait
      if (mode == kModeScramble && gait == kGaitStand) continue;
      bool lower = com_speed > kGaitAuto[gait];
      bool upper = gait == kGaitGallop || com_speed <= kGaitAuto[gait+1];
      bool wait = mju_abs(gait_switch_time_ - data->time) > kAutoGaitMinTime;
      if (lower && upper && wait) {
        parameters[gait_param_id_] = ReinterpretAsDouble(gait);
        gait_switch_time_ = data->time;
      }
    }
  }


  // ---------- handle gait switch, manual or auto ----------
  double gait_selection = parameters[gait_param_id_];
  if (gait_selection != current_gait_) {
    current_gait_ = gait_selection;
    A1Gait gait = GetGait();
    parameters[duty_param_id_] = kGaitParam[gait][0];
    parameters[cadence_param_id_] = kGaitParam[gait][1];
    parameters[amplitude_param_id_] = kGaitParam[gait][2];
    weight[balance_cost_id_] = kGaitParam[gait][3];
    weight[upright_cost_id_] = kGaitParam[gait][4];
    weight[height_cost_id_] = kGaitParam[gait][5];
  }


  // ---------- Walk ----------
  double* goal_pos = data->mocap_pos + 3*goal_mocap_id_;
  if (mode == kModeWalk) {
    double angvel = parameters[ParameterIndex(model, "Walk turn")];
    double speed = parameters[ParameterIndex(model, "Walk speed")];

    // current torso direction
    double* torso_xmat = data->xmat + 9*torso_body_id_;
    double forward[2] = {torso_xmat[0], torso_xmat[3]};
    mju_normalize(forward, 2);
    double leftward[2] = {-forward[1], forward[0]};

    // switching into Walk or parameters changed, reset task state
    if (mode != current_mode_ || angvel_ != angvel || speed_ != speed) {
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
    Walk(goal_pos, time);
  }


  // ---------- Flip ----------
  double* compos = SensorByName(model, data, "torso_subtreecom");
  if (mode == kModeFlip) {
    // switching into Flip, reset task state
    if (mode != current_mode_) {
      // save time
      mode_start_time_ = data->time;

      // save body orientation, ground height
      mju_copy4(orientation_, data->xquat + 4*torso_body_id_);
      ground_ = Ground(model, data, compos);

      // save parameters
      save_weight_ = weight;
      save_gait_switch_ = parameters[gait_switch_param_id_];

      // set parameters
      weight[CostTermByName(model, "Upright")] = 0.2;
      weight[CostTermByName(model, "Height")] = 5;
      weight[CostTermByName(model, "Position")] = 0;
      weight[CostTermByName(model, "Gait")] = 0;
      weight[CostTermByName(model, "Balance")] = 0;
      weight[CostTermByName(model, "Effort")] = 0.005;
      weight[CostTermByName(model, "Posture")] = 0.1;
      parameters[gait_switch_param_id_] = ReinterpretAsDouble(0);
    }

    // time from start of Flip
    double flip_time = data->time - mode_start_time_;

    if (flip_time >= jump_time_ + flight_time_ + land_time_) {
      // Flip ended, back to Quadruped, restore values
      mode = kModeQuadruped;
      weight = save_weight_;
      parameters[gait_switch_param_id_] = save_gait_switch_;
      goal_pos[0] = data->site_xpos[3*head_site_id_ + 0];
      goal_pos[1] = data->site_xpos[3*head_site_id_ + 1];
    }
  }

  // save mode
  current_mode_ = static_cast<A1Mode>(mode);
  last_transition_time_ = data->time;
}

// colors of visualisation elements drawn in ModifyScene()
constexpr float kStepRgba[4] = {0.6, 0.8, 0.2, 1};  // step-height cylinders
constexpr float kHullRgba[4] = {0.4, 0.2, 0.8, 1};  // convex hull
constexpr float kAvgRgba[4] = {0.4, 0.2, 0.8, 1};   // average foot position
constexpr float kCapRgba[4] = {0.3, 0.3, 0.8, 1};   // capture point
constexpr float kPcpRgba[4] = {0.5, 0.5, 0.2, 1};   // projected capture point

// draw task-related geometry in the scene
void QuadrupedFlat::ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {
  // flip target pose
  if (current_mode_ == kModeFlip) {
    double flip_time = data->time - mode_start_time_;
    double* torso_pos = data->xpos + 3*torso_body_id_;
    double pos[3] = {torso_pos[0], torso_pos[1], FlipHeight(flip_time)};
    double quat[4];
    FlipQuat(quat, flip_time);
    double mat[9];
    mju_quat2Mat(mat, quat);
    double size[3] = {0.25, 0.15, 0.05};
    float rgba[4] = {0, 1, 0, 0.5};
    AddGeom(scene, mjGEOM_BOX, size, pos, mat, rgba);

    // don't draw anything else during flip
    return;
  }

  // current foot positions
  double* foot_pos[kNumFoot];
  for (A1Foot foot : kFootAll)
    foot_pos[foot] = data->geom_xpos + 3 * foot_geom_id_[foot];

  // stance and flight positions
  double flight_pos[kNumFoot][3];
  double stance_pos[kNumFoot][3];
  for (A1Foot foot : kFootAll) {  // set to foot horizontal positions
    flight_pos[foot][0] = stance_pos[foot][0] = foot_pos[foot][0];
    flight_pos[foot][1] = stance_pos[foot][1] = foot_pos[foot][1];
  }

  // ground height below feet
  double ground[kNumFoot];
  for (A1Foot foot : kFootAll) {
    ground[foot] = Ground(model, data, foot_pos[foot]);
  }

  // step heights
  A1Gait gait = GetGait();
  double step[kNumFoot];
  FootStep(step, GetPhase(data->time), gait);

  // draw step height
  for (A1Foot foot : kFootAll) {
    stance_pos[foot][2] = kFootRadius + ground[foot];
    if (current_mode_ == kModeBiped) {
      // skip "hands" in biped mode
      bool handstand = ReinterpretAsInt(parameters[biped_type_param_id_]);
      bool front_hand = !handstand && (foot == kFootFL || foot == kFootFR);
      bool back_hand = handstand && (foot == kFootHL || foot == kFootHR);
      if (front_hand || back_hand) continue;
    }
    if (step[foot]) {
      flight_pos[foot][2] = kFootRadius + step[foot] + ground[foot];
      AddConnector(scene, mjGEOM_CYLINDER, kFootRadius,
                   stance_pos[foot], flight_pos[foot], kStepRgba);
    }
  }

  // support polygon (currently unused for cost)
  double polygon[2*kNumFoot];
  for (A1Foot foot : kFootAll) {
    polygon[2*foot] = foot_pos[foot][0];
    polygon[2*foot + 1] = foot_pos[foot][1];
  }
  int hull[kNumFoot];
  int num_hull = Hull2D(hull, kNumFoot, polygon);
  for (int i=0; i < num_hull; i++) {
    int j = (i + 1) % num_hull;
    AddConnector(scene, mjGEOM_CAPSULE, kFootRadius/2,
                 stance_pos[hull[i]], stance_pos[hull[j]], kHullRgba);
  }

  // capture point
  bool is_biped = current_mode_ == kModeBiped;
  double height_goal = is_biped ? kHeightBiped : kHeightQuadruped;
  double fall_time = mju_sqrt(2*height_goal / gravity_);
  double capture[3];
  double* compos = SensorByName(model, data, "torso_subtreecom");
  double* comvel = SensorByName(model, data, "torso_subtreelinvel");
  mju_addScl3(capture, compos, comvel, fall_time);

  // ground under CoM
  double com_ground = Ground(model, data, compos);

  // average foot position
  double feet_pos[3];
  AverageFootPos(feet_pos, foot_pos);
  feet_pos[2] = com_ground;

  double foot_size[3] = {kFootRadius, 0, 0};

  // average foot position
  AddGeom(scene, mjGEOM_SPHERE, foot_size, feet_pos, /*mat=*/nullptr, kAvgRgba);

  // capture point
  capture[2] = com_ground;
  AddGeom(scene, mjGEOM_SPHERE, foot_size, capture, /*mat=*/nullptr, kCapRgba);

  // capture point, projected onto hull
  double pcp2[2];
  NearestInHull(pcp2, capture, polygon, hull, num_hull);
  double pcp[3] = {pcp2[0], pcp2[1], com_ground};
  AddGeom(scene, mjGEOM_SPHERE, foot_size, pcp, /*mat=*/nullptr, kPcpRgba);
}

//  ============  task-state utilities  ============
// save task-related ids
void QuadrupedFlat::Reset(const mjModel* model) {
  // call method from base class
  Task::Reset(model);

  // ----------  task identifiers  ----------
  gait_param_id_ = ParameterIndex(model, "select_Gait");
  gait_switch_param_id_ = ParameterIndex(model, "select_Gait switch");
  flip_dir_param_id_ = ParameterIndex(model, "select_Flip dir");
  biped_type_param_id_ = ParameterIndex(model, "select_Biped type");
  cadence_param_id_ = ParameterIndex(model, "Cadence");
  amplitude_param_id_ = ParameterIndex(model, "Amplitude");
  duty_param_id_ = ParameterIndex(model, "Duty ratio");
  balance_cost_id_ = CostTermByName(model, "Balance");
  upright_cost_id_ = CostTermByName(model, "Upright");
  height_cost_id_ = CostTermByName(model, "Height");

  // ----------  model identifiers  ----------
  torso_body_id_ = mj_name2id(model, mjOBJ_XBODY, "trunk");
  if (torso_body_id_ < 0) mju_error("body 'trunk' not found");

  head_site_id_ = mj_name2id(model, mjOBJ_SITE, "head");
  if (head_site_id_ < 0) mju_error("site 'head' not found");

  int goal_id = mj_name2id(model, mjOBJ_XBODY, "goal");
  if (goal_id < 0) mju_error("body 'goal' not found");

  goal_mocap_id_ = model->body_mocapid[goal_id];
  if (goal_mocap_id_ < 0) mju_error("body 'goal' is not mocap");

  // foot geom ids
  int foot_index = 0;
  for (const char* footname : {"FL", "HL", "FR", "HR"}) {
    int foot_id = mj_name2id(model, mjOBJ_GEOM, footname);
    if (foot_id < 0) mju_error_s("geom '%s' not found", footname);
    foot_geom_id_[foot_index] = foot_id;
    foot_index++;
  }

  // shoulder body ids
  int shoulder_index = 0;
  for (const char* shouldername : {"FL_hip", "HL_hip", "FR_hip", "HR_hip"}) {
    int foot_id = mj_name2id(model, mjOBJ_BODY, shouldername);
    if (foot_id < 0) mju_error_s("body '%s' not found", shouldername);
    shoulder_body_id_[shoulder_index] = foot_id;
    shoulder_index++;
  }

  // ----------  derived kinematic quantities for Flip  ----------
  gravity_ = mju_norm3(model->opt.gravity);
  // velocity at takeoff
  jump_vel_ = mju_sqrt(2*gravity_*(kMaxHeight - kLeapHeight));
  // time in flight phase
  flight_time_ = 2 * jump_vel_ / gravity_;
  // acceleration during jump phase
  jump_acc_ = jump_vel_ * jump_vel_ / (2 * (kLeapHeight - kCrouchHeight));
  // time in crouch sub-phase of jump
  crouch_time_ = mju_sqrt(2 * (kHeightQuadruped-kCrouchHeight) / jump_acc_);
  // time in leap sub-phase of jump
  leap_time_ = jump_vel_ / jump_acc_;
  // jump total time
  jump_time_ = crouch_time_ + leap_time_;
  // velocity at beginning of crouch
  crouch_vel_ = -jump_acc_ * crouch_time_;
  // time of landing phase
  land_time_ = 2 * (kLeapHeight-kHeightQuadruped) / jump_vel_;
  // acceleration during landing
  land_acc_ = jump_vel_ / land_time_;
  // rotational velocity during flight phase (rotates 1.25 pi)
  flight_rot_vel_ = 1.25 * mjPI / flight_time_;
  // rotational velocity at start of leap (rotates 0.5 pi)
  jump_rot_vel_ = mjPI / leap_time_ - flight_rot_vel_;
  // rotational acceleration during leap (rotates 0.5 pi)
  jump_rot_acc_ = (flight_rot_vel_ - jump_rot_vel_) / leap_time_;
  // rotational deceleration during land (rotates 0.25 pi)
  land_rot_acc_ =
      2 * (flight_rot_vel_ * land_time_ - mjPI/4) / (land_time_ * land_time_);
}

// compute average foot position, depending on mode
void QuadrupedFlat::AverageFootPos(double avg_foot_pos[3],
                              double* foot_pos[kNumFoot]) const {
  if (current_mode_ == kModeBiped) {
    int handstand = ReinterpretAsInt(parameters[biped_type_param_id_]);
    if (handstand) {
      mju_add3(avg_foot_pos, foot_pos[kFootFL], foot_pos[kFootFR]);
    } else {
      mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
    }
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.5);
  } else {
    mju_add3(avg_foot_pos, foot_pos[kFootHL], foot_pos[kFootHR]);
    mju_addTo3(avg_foot_pos, foot_pos[kFootFL]);
    mju_addTo3(avg_foot_pos, foot_pos[kFootFR]);
    mju_scl3(avg_foot_pos, avg_foot_pos, 0.25);
  }
}

// return phase as a function of time
double QuadrupedFlat::GetPhase(double time) const {
  return phase_start_ + (time - phase_start_time_) * phase_velocity_;
}

// horizontal Walk trajectory
void QuadrupedFlat::Walk(double pos[2], double time) const {
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

// get gait
QuadrupedFlat::A1Gait QuadrupedFlat::GetGait() const {
  if (current_mode_ == kModeBiped)
    return kGaitTrot;
  return static_cast<A1Gait>(ReinterpretAsInt(parameters[gait_param_id_]));
}

// return normalized target step height
double QuadrupedFlat::StepHeight(double time, double footphase,
                                 double duty_ratio) const {
  double angle = fmod(time + mjPI - footphase, 2*mjPI) - mjPI;
  double value = 0;
  if (duty_ratio < 1) {
    angle *= 0.5 / (1 - duty_ratio);
    value = mju_cos(mju_clip(angle, -mjPI/2, mjPI/2));
  }
  return mju_abs(value) < 1e-6 ? 0.0 : value;
}

// compute target step height for all feet
void QuadrupedFlat::FootStep(double step[kNumFoot], double time,
                             A1Gait gait) const {
  double amplitude = parameters[amplitude_param_id_];
  double duty_ratio = parameters[duty_param_id_];
  for (A1Foot foot : kFootAll) {
    double footphase = 2*mjPI*kGaitPhase[gait][foot];
    step[foot] = amplitude * StepHeight(time, footphase, duty_ratio);
  }
}

// height during flip
double QuadrupedFlat::FlipHeight(double time) const {
  if (time >= jump_time_ + flight_time_ + land_time_) {
    return kHeightQuadruped + ground_;
  }
  double h = 0;
  if (time < jump_time_) {
    h = kHeightQuadruped + time * crouch_vel_ + 0.5 * time * time * jump_acc_;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    h = kLeapHeight + jump_vel_*time - 0.5*9.81*time*time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    h = kLeapHeight - jump_vel_*time + 0.5*land_acc_*time*time;
  }
  return h + ground_;
}

// orientation during flip
//  total rotation = leap + flight + land
//            2*pi = pi/2 + 5*pi/4 + pi/4
void QuadrupedFlat::FlipQuat(double quat[4], double time) const {
  double angle = 0;
  if (time >= jump_time_ + flight_time_ + land_time_) {
    angle = 2*mjPI;
  } else if (time >= crouch_time_ && time < jump_time_) {
    time -= crouch_time_;
    angle = 0.5 * jump_rot_acc_ * time * time + jump_rot_vel_ * time;
  } else if (time >= jump_time_ && time < jump_time_ + flight_time_) {
    time -= jump_time_;
    angle = mjPI/2 + flight_rot_vel_ * time;
  } else if (time >= jump_time_ + flight_time_) {
    time -= jump_time_ + flight_time_;
    angle = 1.75*mjPI + flight_rot_vel_*time - 0.5*land_rot_acc_ * time * time;
  }
  int flip_dir = ReinterpretAsInt(parameters[flip_dir_param_id_]);
  double axis[3] = {0, flip_dir ? 1.0 : -1.0, 0};
  mju_axisAngle2Quat(quat, axis, angle);
  mju_mulQuat(quat, orientation_, quat);
}


// --------------------- Residuals for quadruped task --------------------
//   Number of residuals: 4
//     Residual (0): position_z - average(foot position)_z - height_goal
//     Residual (1): position - goal_position
//     Residual (2): orientation - goal_orientation
//     Residual (3): control
//   Number of parameters: 1
//     Parameter (1): height_goal
// -----------------------------------------------------------------------
void QuadrupedHill::Residual(const mjModel* model, const mjData* data,
                             double* residual) const {
  // ---------- Residual (0) ----------
  // standing height goal
  double height_goal = parameters[0];

  // system's standing height
  double standing_height = SensorByName(model, data, "position")[2];

  // average foot height
  double FRz = SensorByName(model, data, "FR")[2];
  double FLz = SensorByName(model, data, "FL")[2];
  double RRz = SensorByName(model, data, "RR")[2];
  double RLz = SensorByName(model, data, "RL")[2];
  double avg_foot_height = 0.25 * (FRz + FLz + RRz + RLz);

  residual[0] = (standing_height - avg_foot_height) - height_goal;

  // ---------- Residual (1) ----------
  // goal position
  const double* goal_position = data->mocap_pos;

  // system's position
  double* position = SensorByName(model, data, "position");

  // position error
  mju_sub3(residual + 1, position, goal_position);

  // ---------- Residual (2) ----------
  // goal orientation
  double goal_rotmat[9];
  const double* goal_orientation = data->mocap_quat;
  mju_quat2Mat(goal_rotmat, goal_orientation);

  // system's orientation
  double body_rotmat[9];
  double* orientation = SensorByName(model, data, "orientation");
  mju_quat2Mat(body_rotmat, orientation);

  mju_sub(residual + 4, body_rotmat, goal_rotmat, 9);

  // ---------- Residual (3) ----------
  mju_copy(residual + 13, data->ctrl, model->nu);
}

// -------- Transition for quadruped task --------
//   If quadruped is within tolerance of goal ->
//   set goal to next from keyframes.
// -----------------------------------------------
void QuadrupedHill::Transition(const mjModel* model, mjData* data) {
  // set mode to GUI selection
  if (mode > 0) {
    current_mode = mode - 1;
  } else {
    // ---------- Compute tolerance ----------
    // goal position
    const double* goal_position = data->mocap_pos;

    // goal orientation
    const double* goal_orientation = data->mocap_quat;

    // system's position
    double* position = SensorByName(model, data, "position");

    // system's orientation
    double* orientation = SensorByName(model, data, "orientation");

    // position error
    double position_error[3];
    mju_sub3(position_error, position, goal_position);
    double position_error_norm = mju_norm3(position_error);

    // orientation error
    double geodesic_distance =
        1.0 - mju_abs(mju_dot(goal_orientation, orientation, 4));

    // ---------- Check tolerance ----------
    double tolerance = 1.5e-1;
    if (position_error_norm <= tolerance && geodesic_distance <= tolerance) {
      // update task state
      current_mode += 1;
      if (current_mode == model->nkey) {
        current_mode = 0;
      }
    }
  }

  // ---------- Set goal ----------
  mju_copy3(data->mocap_pos, model->key_mpos + 3 * current_mode);
  mju_copy4(data->mocap_quat, model->key_mquat + 4 * current_mode);
}

}  // namespace mjpc
