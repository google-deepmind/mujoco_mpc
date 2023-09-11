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

// // set model parameters
// void SetModelParameters(mjModel* model, const double* parameters, int dim) {
//   // change damping
//   // model->dof_damping[0] = parameters[0];

//   // // change tip site pos
//   // for (int i = 0; i < 3; i++) {
//   //   model->site_pos[i] = parameters[1 + i];
//   // }
//   model->site_sameframe[0] = 0;
//   model->site_pos[0] = parameters[0];
//   model->site_pos[1] = parameters[1];
//   model->site_pos[2] = parameters[2];
// }

// // model parameter jacobians
// void ModelParameterJacobians(double* dfdp, double* dsdp, mjModel* model,
//                              mjData* data, double* parameters,
//                              int dim_parameters, double* scratch,
//                              double eps = 1.0e-6) {
//   // unpack
//   double* dpdf = scratch;  // transposed force Jacobian
//   double* dpds =
//       scratch + dim_parameters * model->nv;  // transposed sensor Jacobian
//   double* f = dpds + dim_parameters *
//                          model->nsensordata;  // nominal force (qfrc_inverse)
//   double* s = f + model->nv;                  // nominal sensors (sensordata)

//   // evaluate nominal parameters
//   SetModelParameters(model, parameters, dim_parameters);

//   // nominal inverse dynamics
//   mj_inverse(model, data);

//   // save nominal force, sensor
//   mju_copy(f, data->qfrc_inverse, model->nv);
//   mju_copy(s, data->sensordata, model->nsensordata);

//   printf("nominal site pos\n");
//   mju_printMat(model->site_pos, 1, 3);

//   // loop over parameters
//   for (int i = 0; i < dim_parameters; i++) {
//     // save nominal parameter
//     double parameter_cache = parameters[i];

//     // nudge parameter i
//     parameters[i] += eps;

//     // set parameters
//     SetModelParameters(model, parameters, dim_parameters);

//     printf("nudged site pos (%i) = \n",i);
//     mju_printMat(model->site_pos, 1, 3);

//     // inverse dynamics
//     // mj_inverse(model, data);
//     mj_forward(model, data);

//     printf("data->xpos = \n");
//     mju_printMat(data->xpos, 1, 3);

//     printf("data->site_xpos = \n");
//     mju_printMat(data->site_xpos, 1, 3);

//     printf("data->qfrc = %f\n", data->qfrc_inverse[0]);
//     printf("f = %f\n", f[0]);

//     printf("data->sensordata = \n");
//     mju_printMat(data->sensordata, 1, model->nsensordata);

//     // force difference
//     double* dpidf = dpdf + i * model->nv;
//     mju_sub(dpidf, data->qfrc_inverse, f, model->nv);

//     // force scaling
//     mju_scl(dpidf, dpidf, 1.0 / eps, model->nv);

//     // sensor difference
//     double* dpids = dpds + i * model->nsensordata;
//     mju_sub(dpids, data->sensordata, s, model->nsensordata);

//     // sensor scaling
//     mju_scl(dpids, dpids, 1.0 / eps, model->nsensordata);

//     // restore parameter i
//     parameters[i] = parameter_cache;
//   }

//   // restore parameters
//   SetModelParameters(model, parameters, dim_parameters);

//   // transpose
//   mju_transpose(dfdp, dpdf, dim_parameters, model->nv);
//   mju_transpose(dsdp, dpds, dim_parameters, model->nsensordata);
// }

// TEST(BatchParameter, ParticleDamping) {
//   printf("test parameter jacobian (finite difference)\n");
//   // load model
//   mjModel* model = LoadTestModel("estimator/particle/task1D_damped.xml");
//   model->opt.enableflags |=
//       mjENBL_INVDISCRETE;  // set discrete inverse dynamics

//   // create data
//   mjData* data = mj_makeData(model);

//   printf("damping value = %f\n", model->dof_damping[0]);

//   // nominal parameters
//   // int dim_parameters = 4;
//   // std::vector<double> parameters(dim_parameters);
//   // parameters[0] = model->dof_damping[0];
//   // parameters[1] = model->site_pos[0];
//   // parameters[2] = model->site_pos[1];
//   // parameters[3] = model->site_pos[2];

//   int dim_parameters = 3;
//   std::vector<double> parameters(dim_parameters);
//   parameters[0] = model->site_pos[0];
//   parameters[1] = model->site_pos[1];
//   parameters[2] = model->site_pos[2];

//   printf("sensor pos = \n");
//   mju_printMat(model->site_pos, 1, 3);

//   // set state
//   data->qpos[0] = 1.0;
//   data->qvel[0] = 0.0;

//   // -- parameters Jacobians -- //

//   // memory
//   std::vector<double> dfdp(model->nv * dim_parameters);
//   std::vector<double> dsdp(model->nsensordata * dim_parameters);
//   std::vector<double> scratch(dim_parameters * model->nv +
//                               dim_parameters * model->nsensordata + model->nv
//                               + model->nsensordata);

//   // evaluate
//   ModelParameterJacobians(dfdp.data(), dsdp.data(), model, data,
//                           parameters.data(), dim_parameters, scratch.data());

//   // results
//   printf("dfdp = \n");
//   mju_printMat(dfdp.data(), model->nv, dim_parameters);
//   // printf("dfdp (scaled)= \n");
//   // mju_scl(dfdp.data(), dfdp.data(), mju_pow(model->opt.timestep, 4),
//   //         model->nv * dim_parameters);
//   // mju_printMat(dfdp.data(), model->nv, dim_parameters);

//   printf("dsdp = \n");
//   mju_printMat(dsdp.data(), model->nsensordata, dim_parameters);
//   // printf("dsdp (scaled)= \n");
//   // mju_scl(dsdp.data() + 2 * dim_parameters, dsdp.data() + 2 *
//   dim_parameters,
//   //         mju_pow(model->opt.timestep, 4), 3 * dim_parameters);
//   // mju_printMat(dsdp.data(), model->nsensordata, dim_parameters);

//   // delete data + model
//   mj_deleteData(data);
//   mj_deleteModel(model);
// }

TEST(BatchParameter, ParticleFramePos) {
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

  for (int t = 0; t < T; t++) {
    printf("q (%i) = %f\n", t, sim.qpos.Get(t)[0]);
  }

  for (int t = 0; t < T; t++) {
    printf("v (%i) = %f\n", t, sim.qvel.Get(t)[0]);
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
  std::fill(estimator.parameter_weight.begin(),
            estimator.parameter_weight.end(), 1.0);

  // initial parameters
  printf("parameters initial = \n");
  mju_printMat(estimator.parameters.data(), 1, 6);

  printf("parameters previous = \n");
  mju_printMat(estimator.parameters_previous.data(), 1, 6);

  printf("measurements initial = \n");
  mju_printMat(estimator.sensor_measurement.Data(), T, model->nsensordata);

  // optimize
  ThreadPool pool(1);
  estimator.Optimize(pool);

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

  printf("prior flag = %i\n", estimator.settings.prior_flag);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
