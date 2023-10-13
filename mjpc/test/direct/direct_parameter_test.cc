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

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <vector>
#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "gtest/gtest.h"
#include "mjpc/direct/direct.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(DirectParameter, ParticleFramePos) {
  // verbose
  bool verbose = false;

  // load model
  mjModel* model = LoadTestModel("estimator/particle/task1D_framepos.xml");
  model->opt.enableflags |=
      mjENBL_INVDISCRETE;  // set discrete inverse dynamics

  // create data
  mjData* data = mj_makeData(model);

  // dimensions
  int nq = model->nq;
  int nv = model->nv;
  int ns = model->nsensordata;

  // ----- rollout ----- //
  int T = 5;
  Simulation sim(model, T);
  double q[1] = {1.0};
  sim.SetState(q, NULL);
  auto controller = [](double* ctrl, double time) {};
  sim.Rollout(controller);

  if (verbose) {
    for (int t = 0; t < T; t++) {
      printf("q (%i) = %f\n", t, sim.qpos.Get(t)[0]);
    }

    for (int t = 0; t < T; t++) {
      printf("v (%i) = %f\n", t, sim.qvel.Get(t)[0]);
    }
  }

  // ----- optimizer ----- //
  Direct optimizer(model, T);

  // set data
  mju_copy(optimizer.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(optimizer.sensor_measurement.Data(), sim.sensor.Data(), ns * T);
  mju_copy(optimizer.force_measurement.Data(), sim.qfrc_actuator.Data(),
           nv * T);
  mju_copy(optimizer.parameters.data(), model->site_pos, 6);
  optimizer.parameters[2] += 0.25;  // perturb site0 z coordinate
  optimizer.parameters[5] -= 0.25;  // perturb site1 z coordinate

  // set process noise
  std::fill(optimizer.noise_process.begin(), optimizer.noise_process.end(),
            1.0);

  // set sensor noise
  std::fill(optimizer.noise_sensor.begin(), optimizer.noise_sensor.end(),
            1.0e-5);

  // settings
  optimizer.settings.verbose_optimize = true;
  optimizer.settings.verbose_cost = true;

  // prior
  mju_copy(optimizer.parameters_previous.data(), model->site_pos, 6);
  std::fill(optimizer.noise_parameter.begin(),
            optimizer.noise_parameter.end(), 1.0);

  if (verbose) {
    // initial parameters
    printf("parameters initial = \n");
    mju_printMat(optimizer.parameters.data(), 1, 6);

    printf("parameters previous = \n");
    mju_printMat(optimizer.parameters_previous.data(), 1, 6);

    printf("measurements initial = \n");
    mju_printMat(optimizer.sensor_measurement.Data(), T, model->nsensordata);
  }

  // optimize
  optimizer.Optimize();

  // test parameter recovery
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(optimizer.parameters[i], model->site_pos[i], 1.0e-5);
  }

  if (verbose) {
    // optimized configurations
    printf("qpos optimized =\n");
    mju_printMat(optimizer.configuration.Data(), T, model->nq);

    printf("qvel optimized =\n");
    mju_printMat(optimizer.velocity.Data(), T, model->nv);

    printf("measurements optimized = \n");
    mju_printMat(optimizer.sensor_prediction.Data(), T, model->nsensordata);

    // optimized parameters
    printf("parameters optimized = \n");
    mju_printMat(optimizer.parameters.data(), 1, 6);
  }

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
