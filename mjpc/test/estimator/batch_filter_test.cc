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
#include "mjpc/estimators/batch.h"
#include "mjpc/test/load.h"
#include "mjpc/test/simulation.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// TEST(BatchFilter, Particle1D) {
//   // load model
//   mjModel* model = LoadTestModel("estimator/particle/task1D.xml");
//   mjData* data = mj_makeData(model);

//   // set home keyframe
//   int home_id = mj_name2id(model, mjOBJ_KEY, "home");
//   if (home_id >= 0) mj_resetDataKeyframe(model, data, home_id);

//   // forward to evaluate sensors 
//   mj_forward(model, data);

//   // ctrl
//   double ctrl[1] = {0.25};

//   // sensor
//   double sensor[2];
//   mju_copy(sensor, data->sensordata, model->nsensordata);

//   printf("qpos (data) = \n");
//   mju_printMat(data->qpos, 1, model->nq);

//   printf("qvel (data) = \n");
//   mju_printMat(data->qvel, 1, model->nv);

//   // dimensions
//   // int nq = model->nq, nv = model->nv, ns = model->nsensordata;

//   // threadpool
//   // ThreadPool pool(1);

//   // batch estimator 
//   Batch filter(model, 3);

//   printf("noise process = \n");
//   mju_printMat(filter.noise_process.data(), 1, filter.DimensionProcess());

//   printf("noise sensor = \n");
//   mju_printMat(filter.noise_sensor.data(), 1, filter.DimensionSensor());

//   printf("qpos (filter) = \n");
//   mju_printMat(filter.state.data(), 1, model->nq);

//   printf("qvel (filter) = \n");
//   mju_printMat(filter.state.data() + model->nq, 1, model->nv);

//   // print times 
//   for (int t = 0; t < filter.ConfigurationLength(); t++) {
//     printf("t = %i -> %f\n", t, filter.times.Get(t)[0]);
//   }

//   // configurations 
//   for (int t = 0; t < filter.ConfigurationLength(); t++) {
//     printf("configuration (%i) = ", t);
//     mju_printMat(filter.configuration.Get(t), 1, model->nq);

//     printf("previous (%i) = \n", t);
//     mju_printMat(filter.configuration_previous.Get(t), 1, model->nq);
//   }

//   // covariance 
//   printf("covariance = \n");
//   mju_printMat(filter.Covariance(), filter.DimensionProcess(),
//                filter.DimensionProcess());

//   // prior weight 
//   printf("prior weight = \n");
//   mju_printMat(filter.weight_prior.data(),
//                model->nq * filter.ConfigurationLength(),
//                model->nq * filter.ConfigurationLength());

//   // update 
//   // double noisy_sensor[2];
//   // noisy_sensor[0] = sensor[0] + 0.001;
//   // noisy_sensor[1] = sensor[1] - 0.001;

//   filter.Update(ctrl, sensor);

//   printf("POST UPDATE:\n");

//   // qpos + ctrl + sensor + force + time
//   for (int t = 0; t < filter.ConfigurationLength(); t++) {
//     printf("qpos (%i) = \n", t);
//     mju_printMat(filter.configuration.Get(t), 1, model->nq);

//     printf("qpos [previous] (%i) = \n", t);
//     mju_printMat(filter.configuration_previous.Get(t), 1, model->nq);

//     printf("ctrl (%i) = \n", t);
//     mju_printMat(filter.ctrl.Get(t), 1, model->nu);

//     printf("sensor [measurement] (%i) = \n", t);
//     mju_printMat(filter.sensor_measurement.Get(t), 1, model->nsensordata);

//     printf("force [measurement] (%i) = \n", t);
//     mju_printMat(filter.force_measurement.Get(t), 1, model->nv);

//     printf("time (%i) = %f\n", t, filter.times.Get(t)[0]);
//     printf("\n");
//   }

//   printf("prior weight = \n");
//   mju_printMat(filter.weight_prior.data(),
//                model->nq * filter.ConfigurationLength(),
//                model->nq * filter.ConfigurationLength());

//   // // optimize 
//   // ThreadPool pool(1);
//   // filter.Optimize(pool);

//   // printf("POST OPTIMIZE: \n");

//   // // qpos + ctrl + sensor + force + time
//   // for (int t = 0; t < filter.ConfigurationLength(); t++) {
//   //   printf("qpos (%i) = \n", t);
//   //   mju_printMat(filter.configuration.Get(t), 1, model->nq);

//   //   printf("qpos [previous] (%i) = \n", t);
//   //   mju_printMat(filter.configuration_previous.Get(t), 1, model->nq);

//   //   printf("ctrl (%i) = \n", t);
//   //   mju_printMat(filter.ctrl.Get(t), 1, model->nu);

//   //   printf("sensor [measurement] (%i) = \n", t);
//   //   mju_printMat(filter.sensor_measurement.Get(t), 1, model->nsensordata);

//   //   printf("sensor [prediction] (%i) = \n", t);
//   //   mju_printMat(filter.sensor_prediction.Get(t), 1, model->nsensordata);

//   //   printf("force [measurement] (%i) = \n", t);
//   //   mju_printMat(filter.force_measurement.Get(t), 1, model->nv);

//   //   printf("force [prediction] (%i) = \n", t);
//   //   mju_printMat(filter.force_prediction.Get(t), 1, model->nv);

//   //   printf("time (%i) = %f\n", t, filter.times.Get(t)[0]);
//   //   printf("\n");
//   // }

//   // // shift 
//   // filter.Shift(1);

//   // printf("POST SHIFT: \n");

//   // // qpos + ctrl + sensor + force + time
//   // for (int t = 0; t < filter.ConfigurationLength() - 1; t++) {
//   //   printf("qpos (%i) = \n", t);
//   //   mju_printMat(filter.configuration.Get(t), 1, model->nq);

//   //   printf("qpos [previous] (%i) = \n", t);
//   //   mju_printMat(filter.configuration_previous.Get(t), 1, model->nq);

//   //   printf("ctrl (%i) = \n", t);
//   //   mju_printMat(filter.ctrl.Get(t), 1, model->nu);

//   //   printf("sensor [measurement] (%i) = \n", t);
//   //   mju_printMat(filter.sensor_measurement.Get(t), 1, model->nsensordata);

//   //   printf("sensor [prediction] (%i) = \n", t);
//   //   mju_printMat(filter.sensor_prediction.Get(t), 1, model->nsensordata);

//   //   printf("force [measurement] (%i) = \n", t);
//   //   mju_printMat(filter.force_measurement.Get(t), 1, model->nv);

//   //   printf("force [prediction] (%i) = \n", t);
//   //   mju_printMat(filter.force_prediction.Get(t), 1, model->nv);

//   //   printf("time (%i) = %f\n", t, filter.times.Get(t)[0]);
//   //   printf("\n");
//   // }

//   // delete data + model
//   mj_deleteData(data);
//   mj_deleteModel(model);
// }

TEST(BatchFilter1, Particle1D) {
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task_timevarying.xml");
  mjData* data = mj_makeData(model);
  model->opt.integrator = mjINT_RK4;

  printf("tolerance = %f\n", model->opt.tolerance);
  printf("iterations = %i\n", model->opt.iterations);

  // set home keyframe
  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) mj_resetDataKeyframe(model, data, home_id);

  // ----- simulate ----- //
  int T = 10;
  Simulation sim(model, T);
  auto controller = [](double* ctrl, double time) {
    ctrl[0] = 2.0 * time + 0.5;
    ctrl[1] = 5.0 * time - 0.123;
  };
  sim.Rollout(controller);

  printf("qpos (sim) = \n");
  mju_printMat(sim.qpos.Data(), 3, model->nq);

  printf("qvel (sim) = \n");
  mju_printMat(sim.qvel.Data(), 3, model->nv);

  printf("qacc (sim) = \n");
  mju_printMat(sim.qacc.Data(), 3, model->nv);

  printf("ctrl (sim) = \n");
  mju_printMat(sim.ctrl.Data(), 3, model->nu);

  printf("qfrc (sim) = \n");
  mju_printMat(sim.qfrc_actuator.Data(), 3, model->nv);

  double v1[2];
  mju_sub(v1, sim.qpos.Get(1), sim.qpos.Get(0), model->nv);
  mju_scl(v1, v1, 1.0 / model->opt.timestep, model->nv);

  double v2[2]; 
  mju_sub(v2, sim.qpos.Get(2), sim.qpos.Get(1), model->nv);
  mju_scl(v2, v2, 1.0 / model->opt.timestep, model->nv);

  double a1[2]; 
  mju_sub(a1, v2, v1, model->nv);
  mju_scl(a1, a1, 1.0 / model->opt.timestep, model->nv);

  printf("v1 = \n");
  mju_printMat(v1, 1, model->nv);
  printf("v2 = \n");
  mju_printMat(v2, 1, model->nv);
  printf("a1 = \n");
  mju_printMat(a1, 1, model->nv);

  // set state 
  mju_copy(data->qpos, sim.qpos.Get(1), model->nq);
  mju_copy(data->qvel, v1, model->nv);
  mju_copy(data->qacc, a1, model->nv);

  // inverse 
  mj_inverse(model, data);

  printf("qfrc = \n"); 
  mju_printMat(data->qfrc_inverse, 1, model->nv);

  // printf("qpos (data) = \n");
  // mju_printMat(data->qpos, 1, model->nq);

  // printf("qvel (data) = \n");
  // mju_printMat(data->qvel, 1, model->nv);

  // // dimensions
  // // int nq = model->nq, nv = model->nv, ns = model->nsensordata;

  // // threadpool
  // // ThreadPool pool(1);

  // // batch estimator 
  // Batch filter(model, 3);

  // printf("noise process = \n");
  // mju_printMat(filter.noise_process.data(), 1, filter.DimensionProcess());

  // printf("noise sensor = \n");
  // mju_printMat(filter.noise_sensor.data(), 1, filter.DimensionSensor());

  // printf("qpos (filter) = \n");
  // mju_printMat(filter.state.data(), 1, model->nq);

  // printf("qvel (filter) = \n");
  // mju_printMat(filter.state.data() + model->nq, 1, model->nv);

  // // print times 
  // for (int t = 0; t < filter.ConfigurationLength(); t++) {
  //   printf("t = %i -> %f\n", t, filter.times.Get(t)[0]);
  // }

  // // configurations 
  // for (int t = 0; t < filter.ConfigurationLength(); t++) {
  //   printf("configuration (%i) = ", t);
  //   mju_printMat(filter.configuration.Get(t), 1, model->nq);

  //   printf("previous (%i) = \n", t);
  //   mju_printMat(filter.configuration_previous.Get(t), 1, model->nq);
  // }

  // // covariance 
  // printf("covariance = \n");
  // mju_printMat(filter.Covariance(), filter.DimensionProcess(),
  //              filter.DimensionProcess());

  // // prior weight 
  // printf("prior weight = \n");
  // mju_printMat(filter.weight_prior.data(),
  //              model->nq * filter.ConfigurationLength(),
  //              model->nq * filter.ConfigurationLength());

  // // update 
  // // double noisy_sensor[2];
  // // noisy_sensor[0] = sensor[0] + 0.001;
  // // noisy_sensor[1] = sensor[1] - 0.001;

  // filter.Update(ctrl, sensor);

  // printf("POST UPDATE:\n");

  // // qpos + ctrl + sensor + force + time
  // for (int t = 0; t < filter.ConfigurationLength(); t++) {
  //   printf("qpos (%i) = \n", t);
  //   mju_printMat(filter.configuration.Get(t), 1, model->nq);

  //   printf("qpos [previous] (%i) = \n", t);
  //   mju_printMat(filter.configuration_previous.Get(t), 1, model->nq);

  //   printf("ctrl (%i) = \n", t);
  //   mju_printMat(filter.ctrl.Get(t), 1, model->nu);

  //   printf("sensor [measurement] (%i) = \n", t);
  //   mju_printMat(filter.sensor_measurement.Get(t), 1, model->nsensordata);

  //   printf("force [measurement] (%i) = \n", t);
  //   mju_printMat(filter.force_measurement.Get(t), 1, model->nv);

  //   printf("time (%i) = %f\n", t, filter.times.Get(t)[0]);
  //   printf("\n");
  // }

  // printf("prior weight = \n");
  // mju_printMat(filter.weight_prior.data(),
  //              model->nq * filter.ConfigurationLength(),
  //              model->nq * filter.ConfigurationLength());

  // // // optimize 
  // // ThreadPool pool(1);
  // // filter.Optimize(pool);

  // // printf("POST OPTIMIZE: \n");

  // // // qpos + ctrl + sensor + force + time
  // // for (int t = 0; t < filter.ConfigurationLength(); t++) {
  // //   printf("qpos (%i) = \n", t);
  // //   mju_printMat(filter.configuration.Get(t), 1, model->nq);

  // //   printf("qpos [previous] (%i) = \n", t);
  // //   mju_printMat(filter.configuration_previous.Get(t), 1, model->nq);

  // //   printf("ctrl (%i) = \n", t);
  // //   mju_printMat(filter.ctrl.Get(t), 1, model->nu);

  // //   printf("sensor [measurement] (%i) = \n", t);
  // //   mju_printMat(filter.sensor_measurement.Get(t), 1, model->nsensordata);

  // //   printf("sensor [prediction] (%i) = \n", t);
  // //   mju_printMat(filter.sensor_prediction.Get(t), 1, model->nsensordata);

  // //   printf("force [measurement] (%i) = \n", t);
  // //   mju_printMat(filter.force_measurement.Get(t), 1, model->nv);

  // //   printf("force [prediction] (%i) = \n", t);
  // //   mju_printMat(filter.force_prediction.Get(t), 1, model->nv);

  // //   printf("time (%i) = %f\n", t, filter.times.Get(t)[0]);
  // //   printf("\n");
  // // }

  // // // shift 
  // // filter.Shift(1);

  // // printf("POST SHIFT: \n");

  // // // qpos + ctrl + sensor + force + time
  // // for (int t = 0; t < filter.ConfigurationLength() - 1; t++) {
  // //   printf("qpos (%i) = \n", t);
  // //   mju_printMat(filter.configuration.Get(t), 1, model->nq);

  // //   printf("qpos [previous] (%i) = \n", t);
  // //   mju_printMat(filter.configuration_previous.Get(t), 1, model->nq);

  // //   printf("ctrl (%i) = \n", t);
  // //   mju_printMat(filter.ctrl.Get(t), 1, model->nu);

  // //   printf("sensor [measurement] (%i) = \n", t);
  // //   mju_printMat(filter.sensor_measurement.Get(t), 1, model->nsensordata);

  // //   printf("sensor [prediction] (%i) = \n", t);
  // //   mju_printMat(filter.sensor_prediction.Get(t), 1, model->nsensordata);

  // //   printf("force [measurement] (%i) = \n", t);
  // //   mju_printMat(filter.force_measurement.Get(t), 1, model->nv);

  // //   printf("force [prediction] (%i) = \n", t);
  // //   mju_printMat(filter.force_prediction.Get(t), 1, model->nv);

  // //   printf("time (%i) = %f\n", t, filter.times.Get(t)[0]);
  // //   printf("\n");
  // // }

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
