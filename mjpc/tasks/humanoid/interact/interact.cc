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

#include "mjpc/tasks/humanoid/interact/interact.h"

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"
#include "mjpc/utilities.h"

namespace mjpc::humanoid {

std::string Interact::XmlPath() const {
  return GetModelPath("humanoid/interact/task.xml");
}
std::string Interact::Name() const { return "Humanoid Interact"; }

// --------------- Helper residual functions ----------------- //
void Interact::ResidualFn::UpResidual(const mjModel* model, const mjData* data,
                                      double* residual, std::string&& name,
                                      int* counter) const {
  name.append("_up");
  const double* up_vector = SensorByName(model, data, name);
  residual[(*counter)++] = mju_abs(up_vector[2] - 1.0);
}

void Interact::ResidualFn::HeadHeightResidual(const mjModel* model,
                                              const mjData* data,
                                              double* residual,
                                              int* counter) const {
  const double head_height = SensorByName(model, data, "head_position")[2];
  residual[(*counter)++] =
      mju_abs(head_height - parameters_[kHeadHeightParameterIndex]);
}

void Interact::ResidualFn::TorsoHeightResidual(const mjModel* model,
                                               const mjData* data,
                                               double* residual,
                                               int* counter) const {
  const double torso_height = SensorByName(model, data, "torso_position")[2];
  residual[(*counter)++] =
      mju_abs(torso_height - parameters_[kTorsoHeightParameterIndex]);
}

void Interact::ResidualFn::KneeFeetXYResidual(const mjModel* model,
                                              const mjData* data,
                                              double* residual,
                                              int* counter) const {
  const double* knee_right = SensorByName(model, data, "knee_right");
  const double* knee_left = SensorByName(model, data, "knee_left");
  const double* foot_right = SensorByName(model, data, "foot_right");
  const double* foot_left = SensorByName(model, data, "foot_left");

  double knee_xy_avg[2] = {0.0};
  mju_add(knee_xy_avg, knee_right, knee_left, 2);
  mju_scl(knee_xy_avg, knee_xy_avg, 0.5, 2);

  double foot_xy_avg[2] = {0.0};
  mju_add(foot_xy_avg, foot_right, foot_left, 2);
  mju_scl(foot_xy_avg, foot_xy_avg, 0.5, 2);

  mju_subFrom(knee_xy_avg, foot_xy_avg, 2);
  residual[(*counter)++] = mju_norm(knee_xy_avg, 2);
}

void Interact::ResidualFn::COMFeetXYResidual(const mjModel* model,
                                             const mjData* data,
                                             double* residual,
                                             int* counter) const {
  double* foot_right = SensorByName(model, data, "foot_right");
  double* foot_left = SensorByName(model, data, "foot_left");
  double* com_position = SensorByName(model, data, "torso_subtreecom");

  double foot_xy_avg[2] = {0.0};
  mju_add(foot_xy_avg, foot_right, foot_left, 2);
  mju_scl(foot_xy_avg, foot_xy_avg, 0.5, 2);

  mju_subFrom(foot_xy_avg, com_position, 2);
  residual[(*counter)++] = mju_norm(foot_xy_avg, 2);
}

void Interact::ResidualFn::FacingDirectionResidual(const mjModel* model,
                                                   const mjData* data,
                                                   double* residual,
                                                   int* counter) const {
  if (residual_keyframe_.facing_target.empty()) {
    residual[(*counter)++] = 0;
    return;
  }

  double* torso_forward = SensorByName(model, data, "torso_forward");
  double* torso_position = mjpc::SensorByName(model, data, "torso_position");

  double target[2] = {0.};
  mju_sub(target, residual_keyframe_.facing_target.data(), torso_position, 2);
  mju_normalize(target, 2);
  mju_subFrom(target, torso_forward, 2);
  residual[(*counter)++] = mju_norm(target, 2);
}

void Interact::ResidualFn::ContactResidual(const mjModel* model,
                                           const mjData* data, double* residual,
                                           int* counter) const {
  for (int i = 0; i < kNumberOfContactPairsInteract; i++) {
    const ContactPair& contact = residual_keyframe_.contact_pairs[i];
    if (contact.body1 != kNotSelectedInteract &&
        contact.body2 != kNotSelectedInteract) {
      double dist[3] = {0.};
      contact.GetDistance(dist, data);
      for (int j = 0; j < 3; j++) residual[(*counter)++] = mju_abs(dist[j]);
    } else {
      for (int j = 0; j < 3; j++) residual[(*counter)++] = 0;
    }
  }
}

void Interact::SaveParamsToKeyframe(ContactKeyframe& kf) const {
  for (int i = 0; i < weight_names.size(); i++) {
    kf.weight[weight_names[i]] = weight[i];
  }
  kf.target_distance_tolerance = parameters[kDistanceToleranceParameterIndex];
  kf.time_limit = parameters[kTimeLimitParameterIndex];
  kf.success_sustain_time = parameters[kSustainTimeParameterIndex];
}

void Interact::LoadParamsFromKeyframe(const ContactKeyframe& kf) {
  ContactKeyframe current_keyframe = motion_strategy_.GetCurrentKeyframe();
  weight.clear();
  int index = 0;
  for (auto& w : weight_names) {
    if (kf.weight.find(w) != kf.weight.end()) {
      weight.push_back(kf.weight.at(w));
    } else {
      double default_weight =
          default_weights[residual_.current_task_mode_][index];
      std::printf(
          "Keyframe %s does not have weight for %s, set default %.1f.\n",
          kf.name.c_str(), w.c_str(), default_weight);
      weight.push_back(default_weight);
    }
    current_keyframe.weight[w] = weight[index];
    index++;
  }
  current_keyframe.name = kf.name;
  parameters[kDistanceToleranceParameterIndex] = kf.target_distance_tolerance;
  parameters[kTimeLimitParameterIndex] = kf.time_limit;
  parameters[kSustainTimeParameterIndex] = kf.success_sustain_time;
}

// ---------------- Residuals for humanoid interaction task ----------- //
//   Number of residuals: 13
//     Residual (0): Torso up
//     Residual (1): Pelvis up
//     Residual (2): Right foot up
//     Residual (3): Left foot up
//     Residual (4): Head height
//     Residual (5): Torso height
//     Residual (6): Knee feet xy-plane distance
//     Residual (7): COM feet xy-plane distance
//     Residual (8): Facing direction target xy-distance
//     Residual (9): Com Vel: should be 0 and equal feet average vel
//     Residual (10): Control: minimise control
//     Residual (11): Joint vel: minimise joint velocity
//     Residual (12): Contact: minimise distance between contact pairs
//   Number of parameters: 2
//     Parameter (0): head_height_goal
//     Parameter (1): torso_height_goal
// -------------------------------------------------------------------- //
void Interact::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;

  double* com_velocity = SensorByName(model, data, "torso_subtreelinvel");

  // ----- task-specific residual terms ------ //
  UpResidual(model, data, residual, "torso", &counter);
  UpResidual(model, data, residual, "pelvis", &counter);
  UpResidual(model, data, residual, "foot_right", &counter);
  UpResidual(model, data, residual, "foot_left", &counter);
  HeadHeightResidual(model, data, residual, &counter);
  TorsoHeightResidual(model, data, residual, &counter);
  KneeFeetXYResidual(model, data, residual, &counter);
  COMFeetXYResidual(model, data, residual, &counter);
  FacingDirectionResidual(model, data, residual, &counter);

  // ----- COM xy velocity regularization ---- //
  mju_copy(&residual[counter], com_velocity, 2);
  counter += 2;

  // ----- joint velocity regularization ----- //
  mju_copy(residual + counter, data->qvel + 6,
           model->nv - (6 + kNumberOfFreeJoints * 6));
  counter += model->nv - (6 + kNumberOfFreeJoints * 6);

  // ----- action regularization ------------- //
  mju_copy(&residual[counter], data->ctrl, model->nu);
  counter += model->nu;

  // ----- contact residual ------------------ //
  ContactResidual(model, data, residual, &counter);

  CheckSensorDim(model, counter);
}

// -------- Transition for interaction task -------- //
void Interact::TransitionLocked(mjModel* model, mjData* data) {
  //  If the task mode is changed, sync it with the residual here
  if (residual_.current_task_mode_ != mode) {
    residual_.current_task_mode_ = (TaskMode)mode;
    weight = default_weights[residual_.current_task_mode_];
  }

  //  If the motion strategy is not initialized, load the given strategy
  if (!motion_strategy_.HasKeyframes()) {
    motion_strategy_.LoadStrategy("armchair_cross_leg");
    LoadParamsFromKeyframe(motion_strategy_.GetCurrentKeyframe());
    return;
  }

  const ContactKeyframe& current_keyframe =
      motion_strategy_.GetCurrentKeyframe();
  const double total_distance = motion_strategy_.CalculateTotalKeyframeDistance(
      data, ContactKeyframeErrorType::kNorm);

  if (data->time - motion_strategy_.GetCurrentKeyframeStartTime() >
          current_keyframe.time_limit &&
      total_distance > current_keyframe.target_distance_tolerance) {
    // timelimit reached but distance criteria not reached, reset
    motion_strategy_.Reset();
    LoadParamsFromKeyframe(motion_strategy_.GetCurrentKeyframe());
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
    motion_strategy_.SetCurrentKeyframeStartTime(data->time);
  } else if (total_distance <= current_keyframe.target_distance_tolerance &&
             data->time -
                     motion_strategy_.GetCurrentKeyframeSuccessStartTime() >
                 current_keyframe.success_sustain_time) {
    // success criteria reached, go to the next keyframe
    motion_strategy_.NextKeyframe();
    LoadParamsFromKeyframe(motion_strategy_.GetCurrentKeyframe());
    residual_.residual_keyframe_ = motion_strategy_.GetCurrentKeyframe();
  } else if (total_distance > current_keyframe.target_distance_tolerance) {
    // keyframe error is more than tolerance, update the success start time
    motion_strategy_.SetCurrentKeyframeSuccessStartTime(data->time);
  }
  SaveParamsToKeyframe(motion_strategy_.GetCurrentKeyframe());
}

// draw task-related geometry in the scene
void Interact::ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {
  // add visuals for the contact points
  mjtNum global_pos[3] = {0.};
  for (int i = 0; i < kNumberOfContactPairsInteract; i++) {
    const ContactPair contact = residual_.residual_keyframe_.contact_pairs[i];
    if (contact.body1 != kNotSelectedInteract) {
      mju_mulMatVec(global_pos, data->xmat + 9 * contact.body1,
                    contact.local_pos1, 3, 3);
      mju_addTo(global_pos, data->xpos + 3 * contact.body1, 3);
      AddGeom(scene, mjGEOM_SPHERE, kVisualPointSize, global_pos, nullptr,
              kContactPairColor[i]);
    }
    if (contact.body2 != kNotSelectedInteract) {
      mju_mulMatVec(global_pos, data->xmat + 9 * contact.body2,
                    contact.local_pos2, 3, 3);
      mju_addTo(global_pos, data->xpos + 3 * contact.body2, 3);
      AddGeom(scene, mjGEOM_SPHERE, kVisualPointSize, global_pos, nullptr,
              kContactPairColor[i]);
    }
  }

  // add visuals for the facing direction
  if (!residual_.residual_keyframe_.facing_target.empty()) {
    mjtNum pos[3] = {residual_.residual_keyframe_.facing_target[0],
                     residual_.residual_keyframe_.facing_target[1], 0};
    AddGeom(scene, mjGEOM_SPHERE, kVisualPointSize, pos, nullptr,
            kFacingDirectionColor);
  }
}

}  // namespace mjpc::humanoid
