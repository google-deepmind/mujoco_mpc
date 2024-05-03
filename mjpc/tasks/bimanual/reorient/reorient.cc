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

#include "mjpc/tasks/bimanual/reorient/reorient.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::aloha {

using absl::Uniform;

std::string Reorient::XmlPath() const {
  return GetModelPath("bimanual/reorient/task.xml");
}
std::string Reorient::Name() const { return "Bimanual Reorient"; }

void Reorient::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;
  double* object_pos = SensorByName(model, data, "object_pos");

  // reach left, encourage proper alignment
  double* left_gripper = SensorByName(model, data, "left/gripper");
  mju_copy3(residual + counter, left_gripper);
  residual[counter + 1] *= 3;
  residual[counter + 2] *= 3;
  counter += 3;

  // reach right, encourage proper alignment
  double* right_gripper = SensorByName(model, data, "right/gripper");
  mju_copy3(residual + counter, right_gripper);
  residual[counter + 1] *= 3;
  residual[counter + 2] *= 3;
  counter += 3;

  // ===== grasp

  // normal arrays, counters
  double normal[4][3] = {{0}, {0}, {0}, {0}};
  int nnormal[4] = {0, 0, 0, 0};

  // get body ids, object id
  int finger[4] = {mj_name2id(model, mjOBJ_BODY, "left/left_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "left/right_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "right/left_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "right/right_finger_link")};
  int object_id = mj_name2id(model, mjOBJ_BODY, "cross");


  int ncon = data->ncon;
  for (int i = 0; i < ncon; ++i) {
    const mjContact* con = data->contact + i;
    int body[2] = {model->geom_bodyid[con->geom[0]],
                   model->geom_bodyid[con->geom[1]]};
    for (int j = 0; j < 2; ++j) {
      if (body[j] == object_id) {
        for (int k = 0; k < 4; ++k) {
          if (body[1-j] == finger[k]) {
            // We want the normal to point from the finger to the object.
            // In mjContact the normal always points from body[0] to body[1].
            // Since body[j] is the object, if j == 0, the normal is flipped.
            double sign = (j == 0) ? -1 : 1;
            mju_addToScl3(normal[k], con->frame, sign);
            nnormal[k]++;
          }
        }
      }
    }
  }

  double grasp = 1;

  // left hand
  if (nnormal[0] && nnormal[1]) {
    mju_normalize3(normal[0]);
    mju_normalize3(normal[1]);
    grasp = 0.5 * (mju_dot3(normal[0], normal[1]) + 1);
  }
  residual[counter++] = grasp;

  // right hand
  grasp = 1;
  if (nnormal[2] && nnormal[3]) {
    mju_normalize3(normal[2]);
    mju_normalize3(normal[3]);
    grasp = 0.5 * (mju_dot3(normal[2], normal[3]) + 1);
  }
  residual[counter++] = grasp;

  // ---------- Bring and match orientation ----------
  double* target_pos = SensorByName(model, data, "target_pos");
  int target_orient_id = mj_name2id(model, mjOBJ_BODY, "target_orient");
  double *target_orient = data->ximat + 9*target_orient_id;
  double *object_orient = data->ximat + 9*object_id;

  // construct crosses
  constexpr double kRadius = 0.05;

  double target_orientT[9];
  mju_transpose(target_orientT, target_orient, 3, 3);
  double object_orientT[9];
  mju_transpose(object_orientT, object_orient, 3, 3);

  double target_cross[18];
  double object_cross[18];
  for (int dim = 0; dim < 3; dim++) {
    for (int side = 0; side < 2; side++) {
      double sign = side ? 1 : -1;
      int offset = 3 * (2*dim + side);
      mju_addScl3(target_cross + offset,
                  target_pos,
                  target_orientT + 3*dim,
                  sign * kRadius);
      mju_addScl3(object_cross + offset,
                  object_pos,
                  object_orientT + 3*dim,
                  sign * kRadius);
    }
  }

  mju_sub(residual + counter, target_cross, object_cross, 18);
  counter += 18;

  CheckSensorDim(model, counter);
}

void Reorient::TransitionLocked(mjModel* model, mjData* data) {
  double residual[26];
  residual_.Residual(model, data, residual);
  double dist = mju_norm(residual + 8, 18);

  // reset target on success
  constexpr int kMinSolveTime = 3;
  if (data->time > last_solve_time + kMinSolveTime && dist < .02) {
    int target_orient_id = mj_name2id(model, mjOBJ_JOINT, "target_orient");
    int dof = model->jnt_dofadr[target_orient_id];

    // random target rotational velocity
    absl::BitGen gen_;
    for (int i = 0; i < 3; ++i) {
      data->qvel[dof + i] = Uniform<double>(gen_, -30, 30);
    }

    // set solve time
    last_solve_time = data->time;
  }
}

}  // namespace mjpc::aloha
