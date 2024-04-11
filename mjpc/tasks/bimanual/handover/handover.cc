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

#include "mjpc/tasks/bimanual/handover/handover.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::aloha {

using absl::Uniform;

std::string Handover::XmlPath() const {
  return GetModelPath("bimanual/handover/task.xml");
}
std::string Handover::Name() const { return "Bimanual Handover"; }

void Handover::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;

  // reach left, encourage proper alignment
  double* left_gripper = SensorByName(model, data, "left/gripper");
  mju_copy3(residual + counter, left_gripper);
  // The sensor is "object pos in the frame of the gripper".
  // X points forward in the gripper frame.
  // By making the distance larger in Y and Z, the gripper is encouraged to
  // orient itself towards the object.
  residual[counter + 1] *= 2;
  residual[counter + 2] *= 2;
  counter += 3;

  // reach right, encourage proper alignment
  double* right_gripper = SensorByName(model, data, "right/gripper");
  mju_copy3(residual + counter, right_gripper);
  residual[counter + 1] *= 2;
  residual[counter + 2] *= 2;
  counter += 3;

  // grasp

  // normal arrays, counters
  double normal[4][3] = {{0}, {0}, {0}, {0}};
  int nnormal[4] = {0, 0, 0, 0};

  // get body ids, object position
  int finger[4] = {mj_name2id(model, mjOBJ_BODY, "left/left_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "left/right_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "right/left_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "right/right_finger_link")};
  int object_id = mj_name2id(model, mjOBJ_BODY, "box");
  double object_pos[3];
  mju_copy3(object_pos, data->xipos + 3*object_id);

  // loop over contacts, add up (and maybe flip) relevant normals
  int ncon = data->ncon;
  for (int i = 0; i < ncon; ++i) {
    const mjContact* con = data->contact + i;
    int bb[2] = {model->geom_bodyid[con->geom[0]],
                 model->geom_bodyid[con->geom[1]]};
    for (int j = 0; j < 2; ++j) {
      if (bb[j] == object_id) {
        for (int k = 0; k < 4; ++k) {
          if (bb[1-j] == finger[k]) {
            double con2object[3];
            mju_sub3(con2object, object_pos, con->pos);
            double sign = mju_dot3(con2object, con->frame) > 0 ? 1 : -1;
            mju_addToScl3(normal[k], con->frame, sign);
            nnormal[k]++;
          }
        }
      }
    }
  }

  // grasp residual
  double grasp = 1;
  double sum[3];

  // left hand
  if (nnormal[0] && nnormal[1]) {
    mju_normalize3(normal[0]);
    mju_normalize3(normal[1]);
    // we want the two normal sums' directions to be opposite each other
    mju_add3(sum, normal[0], normal[1]);
    grasp = 0.5 * mju_norm3(sum);
  }
  residual[counter] = grasp;

  // multiply by right hand
  // the multiplication results in a term that means
  // "one of the hands should grasp the object well"
  if (nnormal[2] && nnormal[3]) {
    mju_normalize3(normal[2]);
    mju_normalize3(normal[3]);
    mju_add3(sum, normal[2], normal[3]);
    grasp = 0.5 * mju_norm3(sum);
    residual[counter] *= grasp;
  }

  // take geometric mean
  residual[counter] = mju_sqrt(mju_max(0, residual[counter]));

  counter++;

  // bring
  double* target = SensorByName(model, data, "target");
  double* box = SensorByName(model, data, "box");
  mju_sub3(residual + counter, box, target);
  counter += 3;

  CheckSensorDim(model, counter);
}

void Handover::TransitionLocked(mjModel* model, mjData* data) {
  double* box = SensorByName(model, data, "box");
  double* target = SensorByName(model, data, "target");
  double vec[3];
  mju_sub3(vec, box, target);
  double dist = mju_norm3(vec);

  // in case user manually reset the env
  if (data->time < last_solve_time) {
    last_solve_time = data->time;
  }

  // reset target on success
  if (data->time > 0 && dist < .01) {
    absl::BitGen gen_;

    // move target
    double flip = target[0] > 0 ? -1 : 1;
    data->mocap_pos[0] = flip * Uniform<double>(gen_, .3, .4);
    double side = Uniform<double>(gen_, 0, 1) > 0.5 ? -1 : 1;
    data->mocap_pos[1] = side * Uniform<double>(gen_, .2, .3);
    data->mocap_pos[2] = Uniform<double>(gen_, 0.25, 0.75);

    // set solve time
    last_solve_time = data->time;
  }

  int nq = model->nq;
  int nv = model->nv;

  // reset box if it falls off table
  if (box[2] < -0.1) {
    // assumes that free body's freejoint is the last joint
    // and that 'home' is the first keyframe
    mju_copy3(data->qpos + nq - 7, model->key_qpos + nq - 7);
    mju_zero3(data->qvel + nv - 6);
  }

  // reset arms if no solution after 30 seconds
  constexpr int kMaxSolveTime = 30;
  if (data->time > last_solve_time + kMaxSolveTime) {
    mju_copy(data->qpos, model->key_qpos, nq);

    // set solve time
    last_solve_time = data->time;
  }
}

}  // namespace mjpc::aloha
