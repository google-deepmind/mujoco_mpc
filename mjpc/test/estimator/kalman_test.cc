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

#include "mjpc/estimators/kalman.h"

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <vector>

#include "gtest/gtest.h"
#include "mjpc/estimators/trajectory.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(Estimator, Kalman) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task_imu.xml");
  mjData* data = mj_makeData(model);

  // dimension
  // int nq = model->nq, nv = model->nv;

  // ----- rollout ----- //
  int T = 200;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {
    // ctrl[0] = mju_sin(10 * time);
    // ctrl[1] = 10 * mju_cos(10 * time);
  };
  double qpos0[1] = {0.25};
  sim.SetState(qpos0, NULL);
  sim.Rollout(controller);

  // ----- Kalman ----- //

  // initialize filter
  Kalman kalman(model);
  kalman.settings.auto_timestep = false;

  // set initial state
  mju_copy(kalman.state.data(), sim.qpos.Get(0), model->nq);
  mju_copy(kalman.state.data() + model->nq, sim.qvel.Get(0), model->nv);

  // initialize covariance
  mju_eye(kalman.covariance.data(), 2 * model->nv);
  mju_scl(kalman.covariance.data(), kalman.covariance.data(), 1.0e-5,
          (2 * model->nv) * (2 * model->nv));

  // initial process noise
  mju_fill(kalman.noise_process.data(), 1.0e-5, 2 * model->nv);

  // initialize sensor noise
  mju_fill(kalman.noise_sensor.data(), 1.0e-5, model->nsensordata);

  // Kalman trajectories
  EstimatorTrajectory<double> kalman_qpos(model->nq, T);
  EstimatorTrajectory<double> kalman_qvel(model->nv, T);
  EstimatorTrajectory<double> kalman_timer_measurement(1, T);
  EstimatorTrajectory<double> kalman_timer_prediction(1, T);

  // noisy sensor
  std::vector<double> noisy_sensor(model->nsensordata);
  absl::BitGen gen_;

  for (int t = 0; t < T; t++) {
    // noisy sensor
    mju_copy(noisy_sensor.data(), sim.sensor.Get(t), model->nsensordata);
    for (int i = 0; i < model->nsensordata; i++) {
      noisy_sensor[i] += 1.0e-1 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }
    // measurement update
    kalman.UpdateMeasurement(sim.ctrl.Get(t), noisy_sensor.data());

    // cache state
    kalman_qpos.Set(kalman.state.data(), t);
    kalman_qvel.Set(kalman.state.data() + model->nq, t);

    // cache timer
    double timer_measurement = kalman.TimerMeasurement();
    kalman_timer_measurement.Set(&timer_measurement, t);

    // prediction update
    kalman.UpdatePrediction();

    // cache timer
    double timer_prediction = kalman.TimerPrediction();
    kalman_timer_prediction.Set(&timer_prediction, t);

    printf("t = %i\n", t);
    printf("  q (sim) = ");
    mju_printMat(sim.qpos.Get(t), 1, model->nq);
    printf("  q (kalman) = ");
    mju_printMat(kalman_qpos.Get(t), 1, model->nq);
    printf("  v (sim) = ");
    mju_printMat(sim.qvel.Get(t), 1, model->nv);
    printf("  v (kalman) = ");
    mju_printMat(kalman_qvel.Get(t), 1, model->nv);
    printf("  timer (measurement) = %.4f\n",
           kalman_timer_measurement.Get(t)[0]);
    printf("  timer (prediction) = %.4f\n", kalman_timer_prediction.Get(t)[0]);
    printf("\n");
  }

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
