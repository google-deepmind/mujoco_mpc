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

#include "mjpc/estimators/unscented.h"

#include <vector>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "gtest/gtest.h"
#include "mjpc/direct/trajectory.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(Unscented, Particle1D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task1D.xml");
  mjData* data = mj_makeData(model);

  // ----- rollout ----- //
  int T = 5;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {};
  double qpos0[1] = {0.25};
  sim.SetState(qpos0, NULL);
  sim.Rollout(controller);

  // ----- Unscented ----- //

  // initialize unscented
  Unscented unscented(model);

  // set initial state
  mju_copy(unscented.state.data(), sim.qpos.Get(0), model->nq);
  mju_copy(unscented.state.data() + model->nq, sim.qvel.Get(0), model->nv);

  // initialize covariance
  mju_eye(unscented.covariance.data(), 2 * model->nv);
  mju_scl(unscented.covariance.data(), unscented.covariance.data(), 1.0e-5,
          (2 * model->nv) * (2 * model->nv));

  // initial process noise
  mju_fill(unscented.noise_process.data(), 1.0e-5, 2 * model->nv);

  // initialize sensor noise
  mju_fill(unscented.noise_sensor.data(), 1.0e-5, model->nsensordata);

  // filter trajectories
  DirectTrajectory<double> unscented_qpos(model->nq, T);
  DirectTrajectory<double> unscented_qvel(model->nv, T);
  DirectTrajectory<double> unscented_timer_update(1, T);

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
    unscented.Update(sim.ctrl.Get(t), noisy_sensor.data());

    // cache state
    unscented_qpos.Set(unscented.state.data(), t);
    unscented_qvel.Set(unscented.state.data() + model->nq, t);

    // cache timer
    double timer_update = unscented.TimerUpdate();
    unscented_timer_update.Set(&timer_update, t);

    // test qpos
    std::vector<double> pos_error(model->nq);
    mju_sub(pos_error.data(), unscented_qpos.Get(t), sim.qpos.Get(t + 1),
            model->nq);
    EXPECT_NEAR(mju_norm(pos_error.data(), model->nq), 0.0, 1.0e-4);

    // test qvel
    std::vector<double> vel_error(model->nv);
    mju_sub(vel_error.data(), unscented_qvel.Get(t), sim.qvel.Get(t + 1),
            model->nv);
    EXPECT_NEAR(mju_norm(vel_error.data(), model->nv), 0.0, 1.0e-4);
  }

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(Unscented, Box3Drot) {
  // load model
  mjModel* model = LoadTestModel("estimator/box/task3Drot.xml");
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

  // ----- Unscented ----- //

  // initialize unscented
  Unscented unscented(model);

  // set initial state
  mju_copy(unscented.state.data(), sim.qpos.Get(0), model->nq);
  mju_copy(unscented.state.data() + model->nq, sim.qvel.Get(0), model->nv);

  // initialize covariance
  mju_eye(unscented.covariance.data(), 2 * model->nv);
  mju_scl(unscented.covariance.data(), unscented.covariance.data(), 1.0e-5,
          (2 * model->nv) * (2 * model->nv));

  // initial process noise
  mju_fill(unscented.noise_process.data(), 1.0e-5, 2 * model->nv);

  // initialize sensor noise
  mju_fill(unscented.noise_sensor.data(), 1.0e-5, model->nsensordata);

  // filter trajectories
  DirectTrajectory<double> unscented_qpos(model->nq, T);
  DirectTrajectory<double> unscented_qvel(model->nv, T);
  DirectTrajectory<double> unscented_timer_update(1, T);

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
    unscented.Update(sim.ctrl.Get(t), noisy_sensor.data());

    // cache state
    unscented_qpos.Set(unscented.state.data(), t);
    unscented_qvel.Set(unscented.state.data() + model->nq, t);

    // cache timer
    double timer_update = unscented.TimerUpdate();
    unscented_timer_update.Set(&timer_update, t);

    // test qpos
    std::vector<double> pos_error(model->nv);
    mju_subQuat(pos_error.data(), unscented_qpos.Get(t), sim.qpos.Get(t + 1));
    EXPECT_NEAR(mju_norm(pos_error.data(), model->nv), 0.0, 1.0e-4);

    // test qvel
    std::vector<double> vel_error(model->nv);
    mju_sub(vel_error.data(), unscented_qvel.Get(t), sim.qvel.Get(t + 1),
            model->nv);
    EXPECT_NEAR(mju_norm(vel_error.data(), model->nv), 0.0, 1.0e-4);
  }

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
