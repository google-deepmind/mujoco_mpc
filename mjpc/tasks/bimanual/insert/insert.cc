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

#include "mjpc/tasks/bimanual/insert/insert.h"

#include <string>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc::aloha {

using absl::Gaussian;

std::string Insert::XmlPath() const {
  return GetModelPath("bimanual/insert/task.xml");
}
std::string Insert::Name() const { return "Bimanual Insert"; }

void Insert::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  int counter = 0;

  // ======== reach

  // reach left
  double* left_gripper = SensorByName(model, data, "left/gripper");
  mju_copy3(residual + counter, left_gripper);
  counter += 3;

  // reach right
  double* right_gripper = SensorByName(model, data, "right/gripper");
  mju_copy3(residual + counter, right_gripper);
  counter += 3;

  // ======== grasp

  // normal arrays, counters
  double normal[4][3] = {{0}, {0}, {0}, {0}};
  int nnormal[4] = {0, 0, 0, 0};

  // get body ids, object position
  int finger[4] = {mj_name2id(model, mjOBJ_BODY, "left/left_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "left/right_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "right/left_finger_link"),
                   mj_name2id(model, mjOBJ_BODY, "right/right_finger_link")};

  int connector[2] = {mj_name2id(model, mjOBJ_GEOM, "connector_f_grip"),
                      mj_name2id(model, mjOBJ_GEOM, "connector_m_grip")};

  // the grasping cost here is different to the one in handover.cc:
  // - contacts between body (finger) and specific geom (connector_x_grip)
  // - left hand always grasps female, right hand always grasps male

  // loop over contacts, add up (and maybe flip) relevant normals
  int ncon = data->ncon;
  for (int i = 0; i < ncon; ++i) {
    const mjContact* con = data->contact + i;
    int bb[2] = {model->geom_bodyid[con->geom[0]],
                 model->geom_bodyid[con->geom[1]]};
    double scale = con->exclude == 0 ? 10 : 1;
    for (int j = 0; j < 2; ++j) {
      if (con->geom[j] == connector[0]) {
        // left hand
        for (int k = 0; k < 2; ++k) {
          if (bb[1-j] == finger[k]) {
            // We want the normal to point from the finger to the object.
            // In mjContact the normal always points from body[0] to body[1].
            // Since body[j] is the object, if j == 0, the normal is flipped.
            double sign = (j == 0) ? -1 : 1;
            mju_addToScl3(normal[k], con->frame, scale * sign);
            nnormal[k]++;
          }
        }
      }

      // right hand
      if (con->geom[j] == connector[1]) {
        for (int k = 2; k < 4; ++k) {
          if (bb[1-j] == finger[k]) {
            double sign = (j == 0) ? -1 : 1;  // as above
            mju_addToScl3(normal[k], con->frame, scale * sign);
            nnormal[k]++;
          }
        }
      }
    }
  }

  // the grasp cost below includes both the contact alignment term and frame
  // alignment of the x-axes of the object and gripper

  double grasp = 1;

  // left hand
  if (nnormal[0] && nnormal[1]) {
    double* left_x = SensorByName(model, data, "left/x");
    double* f_x = SensorByName(model, data, "f/x");
    double frame_misalign = mju_dot3(left_x, f_x);
    mju_normalize3(normal[0]);
    mju_normalize3(normal[1]);
    double con_misalign = mju_dot3(normal[0], normal[1]);
    grasp = (con_misalign + 2*frame_misalign + 3) / 6;
  }
  residual[counter++] = grasp;

  // right hand
  grasp = 1;
  if (nnormal[2] && nnormal[3]) {
    double* right_x = SensorByName(model, data, "right/x");
    double* m_x = SensorByName(model, data, "m/x");
    double frame_misalign = mju_dot3(right_x, m_x);
    mju_normalize3(normal[2]);
    mju_normalize3(normal[3]);
    double con_misalign = mju_dot3(normal[2], normal[3]);
    grasp = (con_misalign + 2*frame_misalign + 3) / 6;
  }
  residual[counter++] = grasp;

  // ======== Lift (don't care much about x,y)
  int msite_id = mj_name2id(model, mjOBJ_SITE, "connector_m");
  int fsite_id = mj_name2id(model, mjOBJ_SITE, "connector_f");
  double* m_pos = data->site_xpos + 3*msite_id;
  double* f_pos = data->site_xpos + 3*fsite_id;
  int target_id = mj_name2id(model, mjOBJ_GEOM, "target");
  double* target_pos = data->geom_xpos + 3*target_id;

  mju_sub3(residual + counter, m_pos, target_pos);
  residual[counter + 1] *= 0.1;  // we care much less about x and y than z
  residual[counter + 2] *= 0.1;
  counter += 3;
  mju_sub3(residual + counter, f_pos, target_pos);
  residual[counter + 1] *= 0.1;
  residual[counter + 2] *= 0.1;
  counter += 3;

  // ======== Insert

  // the insert residual matches both position and orientation by constructing
  // ad-hoc crosses of 6 points at distance kRadius away from the two sites,
  // aligned with the the orientation matrices (site_pos ± kRadius *{x, y, z})

  double* m_mat = data->site_xmat + 9*msite_id;
  double* f_mat = data->site_xmat + 9*fsite_id;

  // construct crosses
  constexpr double kRadius = 0.08;

  double m_matT[9];
  mju_transpose(m_matT, m_mat, 3, 3);
  double f_matT[9];
  mju_transpose(f_matT, f_mat, 3, 3);

  double m_cross[18];
  double f_cross[18];
  for (int dim = 0; dim < 3; dim++) {
    for (int side = 0; side < 2; side++) {
      double sign = side ? 1 : -1;
      int offset = 3 * (2*dim + side);
      mju_addScl3(m_cross + offset,
                  m_pos,
                  m_matT + 3*dim,
                  sign * kRadius);
      mju_addScl3(f_cross + offset,
                  f_pos,
                  f_matT + 3*dim,
                  sign * kRadius);
    }
  }

  mju_sub(residual + counter, m_cross, f_cross, 18);
  counter += 18;

  CheckSensorDim(model, counter);
}

void Insert::TransitionLocked(mjModel* model, mjData* data) {
  double residual[100];
  residual_.Residual(model, data, residual);
  int nr = ResidualSize(model);
  double dist = mju_norm(residual + nr - 18, 18);

  // reset connectors and target on success
  constexpr int kMinSolveTime = 3;
  constexpr double kMinDist = 0.005;
  if (data->time > last_solve_time + kMinSolveTime && dist < kMinDist) {
    absl::BitGen gen_;

    int home = mj_name2id(model, mjOBJ_KEY, "home");
    int nq = model->nq;

    // reset connectors
    for (auto connector : {"connector_f", "connector_m"}) {
      int jntid = mj_name2id(model, mjOBJ_JOINT, connector);
      int qposadr = model->jnt_qposadr[jntid];
      mjtNum* qpos0 = model->key_qpos+home*nq + qposadr;
      mjtNum* qpos = data->qpos + qposadr;
      mju_copy3(qpos, qpos0);
      for (int i = 0; i < 4; ++i) {
        qpos[3+i] = Gaussian<double>(gen_);
      }
    }

    // set solve time
    last_solve_time = data->time;
  }

  // reset if no solution after 60 seconds
  constexpr int kMaxSolveTime = 60;
  if (data->time > last_solve_time + kMaxSolveTime) {
    mju_copy(data->qpos, model->key_qpos, model->nq);

    // set solve time
    last_solve_time = data->time;
  }
}

}  // namespace mjpc::aloha
