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

#ifndef MJPC_PLANNERS_ILQG_BACKWARD_PASS_H_
#define MJPC_PLANNERS_ILQG_BACKWARD_PASS_H_

#include <vector>

#include "mjpc/planners/cost_derivatives.h"
#include "mjpc/planners/ilqg/boxqp.h"
#include "mjpc/planners/ilqg/policy.h"
#include "mjpc/planners/ilqg/settings.h"
#include "mjpc/planners/model_derivatives.h"

namespace mjpc {

enum BackwardPassRegularization : int {
  kControlRegularization = 0,
  kStateControlRegularization,
  kValueRegularization
};

// data and methods to compute iLQG backward pass
class iLQGBackwardPass {
 public:
  // constructor
  iLQGBackwardPass() = default;

  // destructor
  ~iLQGBackwardPass() = default;

  // ----- methods ----- //
  // allocate memory
  void Allocate(int dim_dstate, int dim_action, int T);

  // reset memory to zeros
  void Reset(int dim_dstate, int dim_action, int T);

  // Riccati at one time step
  int RiccatiStep(int n, int m, double mu, const double *Wx, const double *Wxx,
                  const double *At, const double *Bt, const double *cxt,
                  const double *cut, const double *cxxt, const double *cxut,
                  const double *cuut, double *Vxt, double *Vxxt, double *dut,
                  double *Kt, double *dV, double *Qxt, double *Qut,
                  double *Qxxt, double *Qxut, double *Quut, double *scratch,
                  BoxQP &boxqp, const double *action,
                  const double *action_limits, int reg_type, int limits);

  // compute backward pass using Riccati
  int Riccati(iLQGPolicy *p, const ModelDerivatives *md,
              const CostDerivatives *cd, int dim_dstate, int dim_action, int T,
              double reg, BoxQP &boxqp, const double *actions,
              const double *action_limits, const iLQGSettings &settings);

  // scale backward pass regularization
  void ScaleRegularization(double factor, double reg_min, double reg_max);

  // update backward pass regularization
  void UpdateRegularization(double reg_min, double reg_max, double z, double s);

  // ----- members ----- //
  std::vector<double> Vx;   // cost-to-go gradient    (T * dim_dstate)
  std::vector<double> Vxx;  // cost-to-go Hessian     (T * dim_dstate
                            // * dim_dstate)
  double dV[2];             // cost-to-go error       (2)
  std::vector<double> Qx;   // Q state gradient       ((T - 1) * dim_dstate)
  std::vector<double> Qu;   // Q action gradient      ((T - 1) * dim_action)
  std::vector<double> Qxx;  // Q state Hessian        ((T - 1) *
                            // dim_dstate * dim_dstate)
  std::vector<double> Qxu;  // Q state action Hessian ((T - 1 *
                            // dim_dstate * dim_action))
  std::vector<double>
      Quu;  // Q action Hessian       ((T - 1) * dim_action * dim_action)
  std::vector<double> Q_scratch;  // scratch
  double regularization;          // regularization
  double regularization_rate;     // regularization_rate
  double regularization_factor;   // regularization_factor
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ILQG_BACKWARD_PASS_H_
