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

#include "mjpc/tasks/planar_pusher/planar_pusher.h"

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
  std::string PlanarPusher::XmlPath() const {
    return GetModelPath("planar_pusher/task.xml");
  }
  std::string PlanarPusher::Name() const { return "PlanarPusher"; }

// ------- Residuals for planar_pusher task ------
//     load_pos: load should at goal position
// ------------------------------------------
  void PlanarPusher::Residual(const mjModel* model, const mjData* data,
                          double* residual) const {
    int idx = 0;

    // ---------- load_horz ----------
    auto ee_pos = getEEPosition(model, data);
    residual[idx++] = ee_pos(1) - parameters[0];

    // ---------- load_vert ----------
    residual[idx++] = ee_pos(2) + 0.6;

    // ---------- load_ori ----------
    VecDf des_ee_rot(4);
    des_ee_rot.setOnes();
    des_ee_rot *= 0.5;

    VecDf ee_rot(4);
    mju_copy(ee_rot.data(), data->xquat + 4 * (model->nbody - 1), 4);
    residual[idx++] = ee_rot(0) - des_ee_rot(0);
    residual[idx++] = ee_rot(1) - des_ee_rot(1);
    residual[idx++] = ee_rot(2) - des_ee_rot(2);
    residual[idx++] = ee_rot(3) - des_ee_rot(3);

    // ---------- load_vel ----------
    for (int i=0; i<model->nv; ++i)
    {
      residual[idx++] = data->qvel[i];
    }

    // ---------- frc_con ----------
//    for (int i=0; i<model->nv; ++i)
//    {
//      residual[idx++] = 1.0/data->qfrc_constraint[i];
//    }

    // ---------- acc ----------
//    for (int i=0; i<model->nv; ++i)
//    {
////      residual[6+i] = data->qacc[i];
//      residual[6+i] = data->qfrc_smooth[i] + data->qfrc_constraint[i];
//    }

//    static int skip=0;
//    ++skip;
//    if (skip%10000 == 0)
//    {
//      std::cout << skip << " " << "frc residue: ";
//      for (int i=0; i<model->nv; ++i)
//      {
//        std::cout << (int)residual[6+i] << "\t";
//      }
//      std::cout << std::endl;

//      std::cout << skip << " " << "qfrc_smooth: ";
//      for (int i=0; i<model->nv; ++i)
//      {
//        std::cout << (int)data->qfrc_smooth[i] << "\t";
//      }
//      std::cout << "qfrc_constraint: ";
//      for (int i=0; i<model->nv; ++i)
//      {
//        std::cout << (int)data->qfrc_constraint[i] << "\t";
//      }
//      std::cout << std::endl;
//    }


  }

  Vec3f PlanarPusher::getEEPosition(const mjModel *model, const mjData *data) const
  {
    VecDf ee_pos(3);
    mju_copy(ee_pos.data(), data->xpos + 3*(model->nbody-1), 3);

    return ee_pos;
  }

}  // namespace mjpc
