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

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "gtest/gtest.h"
#include "mjpc/estimators/batch.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(BatchParameter, ParticleFramePos) {
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
  int T = 3;
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

  // ----- estimator ----- //
  Batch estimator(model, T);

  // set data
  mju_copy(estimator.configuration.Data(), sim.qpos.Data(), nq * T);
  mju_copy(estimator.sensor_measurement.Data(), sim.sensor.Data(), ns * T);
  mju_copy(estimator.force_measurement.Data(), sim.qfrc_actuator.Data(),
           nv * T);
  mju_copy(estimator.parameters.data(), model->site_pos, 6);
  estimator.parameters[2] += 0.25;  // perturb site0 z coordinate
  estimator.parameters[5] -= 0.25;  // perturb site1 z coordinate

  // set process noise
  std::fill(estimator.noise_process.begin(), estimator.noise_process.end(),
            1.0);

  // set sensor noise
  std::fill(estimator.noise_sensor.begin(), estimator.noise_sensor.end(),
            1.0e-5);

  // settings
  estimator.settings.verbose_optimize = true;
  estimator.settings.verbose_cost = true;

  // prior
  std::vector<double> prior_weights((T * model->nv) * (T * model->nv));
  std::fill(prior_weights.begin(), prior_weights.end(), 0.0);
  estimator.SetPriorWeights(prior_weights.data(), 0.0);
  mju_copy(estimator.parameters_previous.data(), model->site_pos, 6);
  std::fill(estimator.noise_parameter.begin(),
            estimator.noise_parameter.end(), 1.0);

  if (verbose) {
    // initial parameters
    printf("parameters initial = \n");
    mju_printMat(estimator.parameters.data(), 1, 6);

    printf("parameters previous = \n");
    mju_printMat(estimator.parameters_previous.data(), 1, 6);

    printf("measurements initial = \n");
    mju_printMat(estimator.sensor_measurement.Data(), T, model->nsensordata);
  }
  

  // optimize
  ThreadPool pool(1);
  estimator.Optimize(pool);

  // test parameter recovery
  for (int i = 0; i < 6; i++) {
    EXPECT_NEAR(estimator.parameters[i], model->site_pos[i], 1.0e-5);
  }

  if (verbose) {
    // optimized configurations
    printf("qpos optimized =\n");
    mju_printMat(estimator.configuration.Data(), T, model->nq);

    printf("qvel optimized =\n");
    mju_printMat(estimator.velocity.Data(), T, model->nv);

    printf("measurements optimized = \n");
    mju_printMat(estimator.sensor_prediction.Data(), T, model->nsensordata);

    // optimized parameters
    printf("parameters optimized = \n");
    mju_printMat(estimator.parameters.data(), 1, 6);
  }
  
  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
