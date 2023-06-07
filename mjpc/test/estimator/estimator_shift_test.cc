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

#include <vector>

#include "gtest/gtest.h"
#include "mjpc/estimators/buffer.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/estimators/trajectory.h"
#include "mjpc/test/load.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

TEST(BatchShift, Particle2D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // dimensions
  int nq = model->nq, nv = model->nv, nu = model->nu, ns = model->nsensordata;

  // threadpool
  ThreadPool pool(2);

  // ----- simulate ----- //

  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(100 * time);
    ctrl[1] = mju_cos(100 * time);
  };

  // trajectories
  int horizon_buffer = 20;
  Trajectory qpos_buffer(nq, horizon_buffer + 1);
  Trajectory qvel_buffer(nv, horizon_buffer + 1);
  Trajectory qacc_buffer(nv, horizon_buffer);
  Trajectory ctrl_buffer(nu, horizon_buffer);
  Trajectory qfrc_actuator_buffer(nv, horizon_buffer);
  Trajectory sensor_buffer(ns, horizon_buffer + 1);
  Trajectory time_buffer(1, horizon_buffer + 1);

  // reset
  mj_resetData(model, data);

  // rollout
  for (int t = 0; t < horizon_buffer; t++) {
    // time
    time_buffer.Set(&data->time, t);

    // set control
    controller(data->ctrl, data->time);

    // forward computes instantaneous qacc
    mj_forward(model, data);

    // cache
    qpos_buffer.Set(data->qpos, t);
    qvel_buffer.Set(data->qvel, t);
    qacc_buffer.Set(data->qacc, t);
    ctrl_buffer.Set(data->ctrl, t);
    qfrc_actuator_buffer.Set(data->qfrc_actuator, t);
    sensor_buffer.Set(data->sensordata, t);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data);
  }

  // final cache
  qpos_buffer.Set(data->qpos, horizon_buffer);
  qvel_buffer.Set(data->qvel, horizon_buffer);

  time_buffer.Set(&data->time, horizon_buffer);

  mj_forward(model, data);
  sensor_buffer.Set(data->sensordata, horizon_buffer);

  // ----- estimator ----- //
  for (int horizon_estimator = 3; horizon_estimator < 7; horizon_estimator++) {
    // initialize
    Estimator estimator;
    estimator.Initialize(model);
    estimator.SetConfigurationLength(horizon_estimator);

    // copy buffers
    mju_copy(estimator.configuration_.Data(), qpos_buffer.Data(),
             nq * horizon_estimator);
    mju_copy(estimator.configuration_prior_.Data(),
             estimator.configuration_.Data(), nq * horizon_estimator);
    mju_copy(estimator.action_.Data(), ctrl_buffer.Data(),
             nu * (horizon_estimator - 1));
    mju_copy(estimator.force_measurement_.Data(), qfrc_actuator_buffer.Data(),
             nv * (horizon_estimator - 1));
    mju_copy(estimator.sensor_measurement_.Data(), sensor_buffer.Data(),
             ns * (horizon_estimator - 1));
    mju_copy(estimator.time_.Data(), time_buffer.Data(),
             (horizon_estimator - 1));

    // shift
    for (int shift = 0; shift < 5; shift++) {
      // set buffer length
      ctrl_buffer.length_ = (horizon_estimator - 1) + shift;
      sensor_buffer.length_ = (horizon_estimator - 1) + shift;
      time_buffer.length_ = (horizon_estimator - 1) + shift;

      // update estimator trajectories
      estimator.UpdateTrajectories(sensor_buffer, ctrl_buffer, time_buffer);

      // sensor measurement error
      std::vector<double> sensor_error(ns * (horizon_estimator - 1));
      for (int i = 0; i < horizon_estimator - 1; i++) {
        mju_sub(sensor_error.data() + ns * i, sensor_buffer.Get(i + shift),
                estimator.sensor_measurement_.Get(i), ns);
      }
      EXPECT_NEAR(mju_norm(sensor_error.data(), ns * (horizon_estimator - 1)),
                  0.0, 1.0e-4);

      // force measurement error
      std::vector<double> force_error(nv * (horizon_estimator - 1));
      for (int i = 0; i < horizon_estimator - 1; i++) {
        mju_sub(force_error.data() + nv * i,
                qfrc_actuator_buffer.Get(i + shift),
                estimator.force_measurement_.Get(i), nv);
      }
      EXPECT_NEAR(mju_norm(force_error.data(), nv * (horizon_estimator - 1)),
                  0.0, 1.0e-4);

      // configuration error
      std::vector<double> configuration_error(nq * horizon_estimator);
      for (int i = 0; i < horizon_estimator; i++) {
        mju_sub(configuration_error.data() + nq * i, qpos_buffer.Get(i + shift),
                estimator.configuration_.Get(i), nq);
      }
      EXPECT_NEAR(mju_norm(configuration_error.data(), nq * horizon_estimator),
                  0.0, 1.0e-4);

      // time error
      std::vector<double> time_error(horizon_estimator);
      for (int i = 0; i < horizon_estimator - 1; i++) {
        mju_sub(time_error.data() + i, time_buffer.Get(i + shift),
                estimator.time_.Get(i), 1);
      }
      EXPECT_NEAR(mju_norm(time_error.data(), horizon_estimator - 1), 0.0, 1.0e-4);
    }
  }

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

// TODO(taylor): basically same tests as above--consolidate
TEST(BatchReuse, Particle2D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // dimensions
  int nq = model->nq, nv = model->nv, nu = model->nu, ns = model->nsensordata;

  // threadpool
  ThreadPool pool(2);

  // ----- simulate ----- //

  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(100 * time);
    ctrl[1] = mju_cos(100 * time);
  };

  // trajectories
  int horizon_buffer = 25;
  Trajectory qpos_buffer(nq, horizon_buffer + 1);
  Trajectory qvel_buffer(nv, horizon_buffer + 1);
  Trajectory qacc_buffer(nv, horizon_buffer);
  Trajectory ctrl_buffer(nu, horizon_buffer);
  Trajectory qfrc_actuator_buffer(nv, horizon_buffer);
  Trajectory sensor_buffer(ns, horizon_buffer + 1);
  Trajectory time_buffer(1, horizon_buffer + 1);

  // reset
  mj_resetData(model, data);

  // rollout
  for (int t = 0; t < horizon_buffer; t++) {
    // time
    time_buffer.Set(&data->time, t);

    // set control
    controller(data->ctrl, data->time);

    // forward computes instantaneous qacc
    mj_forward(model, data);

    // cache
    qpos_buffer.Set(data->qpos, t);
    qvel_buffer.Set(data->qvel, t);
    qacc_buffer.Set(data->qacc, t);
    ctrl_buffer.Set(data->ctrl, t);
    qfrc_actuator_buffer.Set(data->qfrc_actuator, t);
    sensor_buffer.Set(data->sensordata, t);

    // step using mj_Euler since mj_forward has been called
    // see mj_ step implementation here
    // https://github.com/deepmind/mujoco/blob/main/src/engine/engine_forward.c#L831
    mj_Euler(model, data);
  }

  // final cache
  qpos_buffer.Set(data->qpos, horizon_buffer);
  qvel_buffer.Set(data->qvel, horizon_buffer);

  time_buffer.Set(&data->time, horizon_buffer);

  mj_forward(model, data);
  sensor_buffer.Set(data->sensordata, horizon_buffer);

  // noisy sensors 
  for (int i = 0; i < ns * (horizon_buffer + 1); i++) {
    absl::BitGen gen_;
    sensor_buffer.Data()[i] += 0.05 * absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // ----- estimator ----- //
  int horizon_estimator = 6;

  // initialize
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(horizon_estimator);

  // copy buffers
  for (int t = 0; t < horizon_estimator; t++) {
    estimator.configuration_.Set(qpos_buffer.Get(t), t);
    estimator.configuration_prior_.Set(qpos_buffer.Get(t), t);
    
    if (t >= horizon_estimator - 1) continue;

    estimator.action_.Set(ctrl_buffer.Get(t), t);
    estimator.force_measurement_.Set(qfrc_actuator_buffer.Get(t), t);
    estimator.sensor_measurement_.Set(sensor_buffer.Get(t), t);
    estimator.time_.Set(time_buffer.Get(t), t);
  }

  // set shift
  int shift = 0;

  for (int i = 0; i < horizon_estimator - 1; i++) {
    // times 
    EXPECT_NEAR(time_buffer.Get(i)[0] - estimator.time_.Get(i)[0], 0.0, 1.0e-5);

    // sensor 
    std::vector<double> error_sensor(ns);
    mju_sub(error_sensor.data(), sensor_buffer.Get(i), estimator.sensor_measurement_.Get(i), ns);
    EXPECT_NEAR(mju_norm(error_sensor.data(), ns), 0.0, 1.0e-5);

    // force 
    std::vector<double> error_force(nv);
    mju_sub(error_force.data(), qfrc_actuator_buffer.Get(i), estimator.force_measurement_.Get(i), nv);
    EXPECT_NEAR(mju_norm(error_force.data(), nv), 0.0, 1.0e-5);
  }

  // shift
  shift++;

  // set buffer length
  ctrl_buffer.length_ = horizon_estimator;
  sensor_buffer.length_ = horizon_estimator;
  time_buffer.length_ = horizon_estimator;

  // update estimator trajectories
  estimator.UpdateTrajectories(sensor_buffer, ctrl_buffer, time_buffer);

  for (int i = 0; i < horizon_estimator - 1; i++) {
    // times 
    EXPECT_NEAR(time_buffer.Get(i + 1)[0] - estimator.time_.Get(i)[0], 0.0, 1.0e-5);

    // sensor 
    std::vector<double> error_sensor(ns);
    mju_sub(error_sensor.data(), sensor_buffer.Get(i + 1), estimator.sensor_measurement_.Get(i), ns);
    EXPECT_NEAR(mju_norm(error_sensor.data(), ns), 0.0, 1.0e-5);

    // force 
    std::vector<double> error_force(nv);
    mju_sub(error_force.data(), qfrc_actuator_buffer.Get(i + 1), estimator.force_measurement_.Get(i), nv);
    EXPECT_NEAR(mju_norm(error_force.data(), nv), 0.0, 1.0e-5);
  }

  // shift
  shift += 2;

  // set buffer length
  ctrl_buffer.length_ = horizon_estimator + 2;
  sensor_buffer.length_ = horizon_estimator + 2;
  time_buffer.length_ = horizon_estimator + 2;

  // update estimator trajectories
  estimator.UpdateTrajectories(sensor_buffer, ctrl_buffer, time_buffer);

  for (int i = 0; i < horizon_estimator - 1; i++) {
    // times 
    EXPECT_NEAR(time_buffer.Get(i + 3)[0] - estimator.time_.Get(i)[0], 0.0, 1.0e-5);

    // sensor 
    std::vector<double> error_sensor(ns);
    mju_sub(error_sensor.data(), sensor_buffer.Get(i + 3), estimator.sensor_measurement_.Get(i), ns);
    EXPECT_NEAR(mju_norm(error_sensor.data(), ns), 0.0, 1.0e-5);

    // force 
    std::vector<double> error_force(nv);
    mju_sub(error_force.data(), qfrc_actuator_buffer.Get(i + 3), estimator.force_measurement_.Get(i), nv);
    EXPECT_NEAR(mju_norm(error_force.data(), nv), 0.0, 1.0e-5);
  }
    
  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(Buffer, Particle2D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task.xml");
  mjData* data = mj_makeData(model);

  // threadpool 
  ThreadPool pool(2);

  // ----- estimator ----- //
  int horizon_estimator = 16;

  // initialize
  Estimator estimator;
  estimator.Initialize(model);
  estimator.SetConfigurationLength(horizon_estimator);
  estimator.verbose_optimize_ = false;

  // ----- simulate ----- //

  // buffer 
  Buffer buffer(model, 32);

  // controller
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = mju_sin(100 * time);
    ctrl[1] = mju_cos(100 * time);
  };

  // reset
  mj_resetData(model, data);

  // rollout
  for (int t = 0; t < 20; t++) {
    printf("t = %i\n", t);

    // set control
    controller(data->ctrl, data->time);

    // step 
    mj_step(model, data);

    // add noise to sensors 
    for (int i = 0; i < model->nsensordata; i++) {
      absl::BitGen gen_;
      data->sensordata[i] += 0.05 * absl::Gaussian<double>(gen_, 0.0, 1.0);
    }

    // update buffer 
    buffer.Update(model, data);

    // update estimator 
    estimator.Update(buffer, pool);

    printf("  cost = %.4f [initial = %.4f]\n", estimator.cost_,
           estimator.cost_initial_);
    printf("  iterations = %i\n", estimator.iterations_smoother_);
  }

  // show buffer 
  // buffer.Print();

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
