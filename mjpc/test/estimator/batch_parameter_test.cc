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

// set model parameters
void SetModelParameters(mjModel* model, const double* parameters, int dim) {
  // change damping
  model->dof_damping[0] = parameters[0];
}

// model parameter jacobians
void ModelParameterJacobians(double* dfdp, double* dsdp, mjModel* model,
                             mjData* data, double* parameters,
                             int dim_parameters, double* scratch,
                             double eps = 1.0e-7) {
  // unpack
  double* dpdf = scratch;  // transposed force Jacobian
  double* dpds =
      scratch + dim_parameters * model->nv;  // transposed sensor Jacobian
  double* f = dpds + dim_parameters *
                         model->nsensordata;  // nominal force (qfrc_inverse)
  double* s = f + model->nv;                  // nominal sensors (sensordata)

  // evaluate nominal parameters
  SetModelParameters(model, parameters, dim_parameters);
  mju_copy(f, data->qfrc_inverse, model->nv);
  mju_copy(s, data->sensordata, model->nsensordata);

  // loop over parameters
  for (int i = 0; i < dim_parameters; i++) {
    // save nominal parameter
    double parameter_cache = parameters[i];

    // nudge parameter i
    parameters[i] += eps;

    // set parameters
    SetModelParameters(model, parameters, dim_parameters);

    // inverse dynamics
    mj_inverse(model, data);

    printf("data->qfrc = %f\n", data->qfrc_inverse[0]);
    printf("f = %f\n", f[0]);

    // force difference
    double* dpidf = dpdf + i * model->nv;
    mju_sub(dpidf, data->qfrc_inverse, f, model->nv);

    // force scaling
    mju_scl(dpidf, dpidf, 1.0 / eps, model->nv);

    // sensor difference
    double* dpids = dpds + i * model->nsensordata;
    mju_sub(dpids, data->sensordata, s, model->nsensordata);

    // sensor scaling
    mju_scl(dpids, dpids, 1.0 / eps, model->nsensordata);

    // restore parameter i
    parameters[i] = parameter_cache;
  }

  // transpose
  mju_transpose(dfdp, dpdf, dim_parameters, model->nv);
  mju_transpose(dsdp, dpds, dim_parameters, model->nsensordata);
}

TEST(BatchParameter, ParticleDamping) {
  printf("test parameter jacobian (finite difference)\n");
  // load model
  mjModel* model = LoadTestModel("estimator/particle/task1D_damped.xml");
  model->opt.enableflags |=
      mjENBL_INVDISCRETE;  // set discrete inverse dynamics

  // create data
  mjData* data = mj_makeData(model);

  printf("damping value = %f\n", model->dof_damping[0]);

  // nominal parameters
  int dim_parameters = 1;
  std::vector<double> parameters(dim_parameters);
  parameters[0] = model->dof_damping[0];

  // -- parameters Jacobians -- //

  // memory
  std::vector<double> dfdp(model->nv * dim_parameters);
  std::vector<double> dsdp(model->nsensordata * dim_parameters);
  std::vector<double> scratch(dim_parameters * model->nv +
                              dim_parameters * model->nsensordata + model->nv +
                              model->nsensordata);

  // evaluate
  ModelParameterJacobians(dfdp.data(), dsdp.data(), model, data,
                          parameters.data(), dim_parameters, scratch.data());

  // results
  printf("dfdp = \n");
  mju_printMat(dfdp.data(), model->nv, dim_parameters);
  printf("dfdp (scaled)= \n");
  mju_scl(dfdp.data(), dfdp.data(), mju_pow(model->opt.timestep, 4),
          model->nv * dim_parameters);
  mju_printMat(dfdp.data(), model->nv, dim_parameters);

  printf("dsdp = \n");
  mju_printMat(dsdp.data(), model->nsensordata, dim_parameters);
  printf("dsdp (scaled)= \n");
  mju_scl(dsdp.data() + 2 * dim_parameters, dsdp.data() + 2 * dim_parameters,
          mju_pow(model->opt.timestep, 4), 3 * dim_parameters);
  mju_printMat(dsdp.data(), model->nsensordata, dim_parameters);

  // delete data + model
  mj_deleteData(data);
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
