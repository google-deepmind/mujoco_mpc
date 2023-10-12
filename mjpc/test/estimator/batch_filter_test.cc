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

#include <cstddef>
#include <vector>

#include "gtest/gtest.h"
#include "mjpc/direct/trajectory.h"
#include "mjpc/estimators/batch.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(BatchFilter, Box3Drot) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task3Drot2.xml");
  mjData* data = mj_makeData(model);

  // ----- rollout ----- //
  int T = 100;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  double qvel[3];
  qvel[0] = 1.0;
  qvel[1] = -0.75;
  qvel[2] = 1.25;
  sim.SetState(NULL, qvel);
  sim.Rollout(controller);

  // ----- Batch ----- //

  // initialize batch
  Batch batch(1);
  batch.settings.time_scaling_force = false;
  batch.settings.time_scaling_sensor = false;
  batch.Initialize(model);
  batch.Reset();

  // set initial state
  mju_copy(batch.state.data(), sim.qpos.Get(0), model->nq);
  mju_copy(batch.state.data() + model->nq, sim.qvel.Get(0), model->nv);

  // set initial configurations
  double* q0 = batch.configuration.Get(0);
  double* q1 = batch.configuration.Get(1);

  mju_copy(q1, sim.qpos.Get(0), model->nq);
  mju_copy(q0, q1, model->nq);
  mj_integratePos(model, q0, sim.qvel.Get(0), -1.0 * model->opt.timestep);

  // initialize covariance
  mju_eye(batch.covariance.data(), 2 * model->nv);
  mju_scl(batch.covariance.data(), batch.covariance.data(), 1.0e-4,
          (2 * model->nv) * (2 * model->nv));

  // initial process noise
  mju_fill(batch.noise_process.data(), 1.0e-4, 2 * model->nv);

  // initialize sensor noise
  mju_fill(batch.noise_sensor.data(), 1.0e-4, model->nsensordata);

  // filter trajectories
  DirectTrajectory<double> batch_qpos(model->nq, T);
  DirectTrajectory<double> batch_qvel(model->nv, T);
  DirectTrajectory<double> batch_timer_update(1, T);

  // noisy sensor
  std::vector<double> noisy_sensor(model->nsensordata);
  absl::BitGen gen_;

  for (int t = 0; t < T - 1; t++) {
    // noisy sensor
    mju_copy(noisy_sensor.data(), sim.sensor.Get(t), model->nsensordata);
    for (int i = 0; i < model->nsensordata; i++) {
      noisy_sensor[i] += 0.0 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    // update
    batch.Update(sim.ctrl.Get(t), noisy_sensor.data());

    // cache state
    batch_qpos.Set(batch.state.data(), t);
    batch_qvel.Set(batch.state.data() + model->nq, t);

    // test qpos
    std::vector<double> pos_error(model->nv);
    mju_subQuat(pos_error.data(), batch_qpos.Get(t), sim.qpos.Get(t + 1));
    EXPECT_NEAR(mju_norm(pos_error.data(), model->nv), 0.0, 5.0e-3);

    // test qvel
    std::vector<double> vel_error(model->nv);
    mju_sub(vel_error.data(), batch_qvel.Get(t), sim.qvel.Get(t + 1),
            model->nv);
    EXPECT_NEAR(mju_norm(vel_error.data(), model->nv), 0.0, 5.0e-3);
  }

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
