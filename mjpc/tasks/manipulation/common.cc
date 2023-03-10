// Copyright 2023 DeepMind Technologies Limited
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

#include "mjpc/tasks/manipulation/common.h"

#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc::manipulation {
namespace {
struct GraspState {
  bool correct_object;
  // maximum distance from a gripper pad to the body, in units of gripper
  // span. populated if correct_object == true
  double gripper_object_distance = 0;
};

GraspState GetGraspState(const mjModel* m, const mjData* d,
                         const ModelValues& model_vals, int body_id) {
  // cast a ray from one finger to the other to decide if the right body is in
  // the gripper
  const double* left_finger =
      d->geom_xpos + 3 * model_vals.left_finger_pad_geom_id;
  const double* right_finger =
      d->geom_xpos + 3 * model_vals.right_finger_pad_geom_id;
  double direction[3];
  mju_sub3(direction, left_finger, right_finger);
  double finger_distance = mju_normalize3(direction);

  // ignore geoms belonging to the robot
  mjtByte geomgroup[mjNGROUP] = {0};
  geomgroup[kGroupCollisionScene] = 1;
  int ray_geom_id;
  double ray_distance =
      mj_ray(m, d, right_finger, direction, geomgroup, 0, -1, &ray_geom_id);

  if (ray_distance == -1 || ray_distance > finger_distance) {
    // nothing in the gripper
    return {.correct_object = false};
  }
  int ray_body_id = m->body_weldid[m->geom_bodyid[ray_geom_id]];
  if (ray_body_id != body_id) {
    // wrong body in the gripper
    return {.correct_object = false};
  }

  // the right object is in the gripper. cast a ray going the opposite way,
  // to see if it's the only object
  mju_scl3(direction, direction, -1);
  double ray_distance2 =
      mj_ray(m, d, left_finger, direction, geomgroup, 0, -1, &ray_geom_id);
  ray_body_id = m->body_weldid[m->geom_bodyid[ray_geom_id]];
  if (ray_body_id != body_id) {
    // there's another object in the gripper
    return {.correct_object = false};
  }

  // return the distance from the gripper to the object, as a fraction of
  // the finger distance
  return {
      .correct_object = true,
      .gripper_object_distance =
          (ray_distance + ray_distance2) / finger_distance,
  };
}

double GripperCost(const mjModel* m, const mjData* d,
                   const ModelValues& model_vals, bool open) {
  double desired_ctrl =
      open ? model_vals.open_gripper_ctrl : model_vals.closed_gripper_ctrl;

  return (d->ctrl[model_vals.gripper_actuator] - desired_ctrl) /
         model_vals.gripper_ctrl_range;
}

}  // namespace

ModelValues ModelValues::FromModel(const mjModel* model) {
  ModelValues values;
  values.robot_body_id = mj_name2id(model, mjOBJ_BODY, "link0");
  values.gripper_site_id = mj_name2id(model, mjOBJ_SITE, "pinch");

  for (const char* name :
       {"right_pad1", "right_pad2", "left_pad1", "left_pad2"}) {
    values.gripper_pad_geoms.push_back(mj_name2id(model, mjOBJ_GEOM, name));
  }
  values.left_finger_pad_geom_id =
      mj_name2id(model, mjOBJ_GEOM, "left_pad2");
  values.right_finger_pad_geom_id =
      mj_name2id(model, mjOBJ_GEOM, "right_pad2");

  values.gripper_actuator = mj_name2id(model, mjOBJ_ACTUATOR, "fingers");
  values.open_gripper_ctrl =
      model->actuator_ctrlrange[2 * values.gripper_actuator];
  values.closed_gripper_ctrl =
      model->actuator_ctrlrange[2 * values.gripper_actuator + 1];
  values.gripper_ctrl_range =
      values.closed_gripper_ctrl - values.open_gripper_ctrl;

  int target_mocap_body_id = mj_name2id(model, mjOBJ_BODY, "target_mocap");
  if (target_mocap_body_id != -1) {
    // this mocap body isn't present in all manipulation models
    values.target_mocap_body = model->body_mocapid[target_mocap_body_id];
  }
  return values;
}

int ComputeControlCost(const mjModel* model, const mjData* data,
                       double* residual) {
  for (int i = 0; i < model->nu; i++) {
    residual[i] =
        data->actuator_force[i] / model->actuator_gainprm[i * mjNGAIN];
  }
  return model->nu;
}

int ComputeGripperVelocityCost(const mjModel* model, const mjData* data,
                               const ModelValues& model_vals,
                               const double* object_linvel, double* residual) {
  double gripper_linvel[6];
  mj_objectVelocity(model, data, mjOBJ_SITE, model_vals.gripper_site_id,
                    gripper_linvel,
                    /*flg_local=*/0);
  // mj_objectVelocity returns angular followed by linear velocity
  mju_sub3(residual, object_linvel, gripper_linvel + 3);
  return 3;
}

namespace {
// returns a length used to scale angular velocities to linear velocities
double BodyExtent(const mjModel* model, int body_id) {
  return mju_sqrt(model->body_inertia[3*body_id] / model->body_mass[body_id]);
}
}  // namespace

int ComputeMaxVelocityCost(const mjModel* model, const mjData* data,
                           const std::vector<int>& body_ids, double* residual) {
  double max_speed = 0;
  mju_zero(residual, 6);
  for (const auto& object : body_ids) {
    double object_vel[6];
    mj_objectVelocity(model, data, mjOBJ_BODY, object, object_vel,
                      /*flg_local=*/0);
    // scale the angular velocity component by the size of the body.
    double extent = BodyExtent(model, object);
    mju_scl3(object_vel, object_vel, extent);
    double speed = mju_norm(object_vel, 6);
    if (speed > max_speed) {
      mju_copy(residual, object_vel, 6);
      max_speed = speed;
    }
  }
  return 6;
}

// returns 1 if there's a body other than exclude_body_id between pos and
// object
double BlockedCost(const mjModel* model, const mjData* data,
                   const ModelValues& model_vals, const double* pos,
                   const double* object, int exclude_body_id) {
  // compute ray direction between pos and object
  double direction[3];
  mju_sub3(direction, object, pos);

  // if pos and object are sufficiently close, don't bother casting a ray
  double distance = mju_normalize3(direction);
  if (distance < 0.02) {
    return 0;
  }

  // ignore geoms belonging to the robot
  mjtByte geomgroup[mjNGROUP] = {0};
  geomgroup[kGroupCollisionScene] = 1;

  int ray_geom_id;
  double ray_distance = mj_ray(model, data, pos, direction, geomgroup, 0,
                               exclude_body_id, &ray_geom_id);
  // if there is something between the hand and the object, return 1.
  return ray_distance > -1 && ray_distance < distance;
}

double GraspQualityCost(const mjModel* m, const mjData* d,
                    const ModelValues& model_vals, int body_id, bool release) {
  if (release) {
    return GripperCost(m, d, model_vals, /*open=*/true);
  }
  GraspState grasp = GetGraspState(m, d, model_vals, body_id);
  constexpr double kEmptyCost = 1;
  if (!grasp.correct_object) {
    // nothing in the gripper, or the wrong object
    return kEmptyCost + GripperCost(m, d, model_vals, /*open=*/true);
  }
  // return the max distance from the gripper to the object, to encourage
  // closing the gripper around the object. gripper_object_distance is 0 to 1
  return grasp.gripper_object_distance;
}

double CarefulCost(const mjModel* model, const mjData* data,
                   const ModelValues& model_vals, int object_body_id) {
  double result = 0;
  for (int i = 0; i < data->ncon; i++) {
    int b1 = model->geom_bodyid[data->contact[i].geom1];
    int b2 = model->geom_bodyid[data->contact[i].geom2];
    int r1 = model->body_rootid[b1];
    int r2 = model->body_rootid[b2];
    if (r1 == model_vals.robot_body_id || r2 == model_vals.robot_body_id) {
      // contact with robot
      if (r2 == object_body_id || r1 == object_body_id) {
        continue;  // contact with the object is okay
      }
      mjtNum force[6];
      mj_contactForce(model, data, i, force);
      result += mju_norm3(force);
    }
  }
  return mju_log10(result + 1);
}

void ComputeRobotiqHandPos(const mjModel* model, const mjData* data,
                    const ModelValues& model_vals, double* pos) {
  for (int geom_id : model_vals.gripper_pad_geoms) {
    mju_addToScl3(pos, data->geom_xpos + 3 * geom_id, 0.25);
  }
}
}  // namespace mjpc::manipulation
