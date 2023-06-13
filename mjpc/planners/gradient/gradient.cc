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

#include "mjpc/planners/gradient/gradient.h"

#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/planners/cost_derivatives.h"
#include "mjpc/planners/gradient/policy.h"
#include "mjpc/planners/model_derivatives.h"
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void Gradient::Allocate(int dim_state_derivative, int dim_action, int T) {
  Vx.resize(dim_state_derivative * T);
  Qx.resize(dim_state_derivative * (T - 1));
  Qu.resize(dim_action * (T - 1));
}

// reset memory to zeros
void Gradient::Reset(int dim_state_derivative, int dim_action, int T) {
  std::fill(Vx.begin(), Vx.begin() + T * dim_state_derivative, 0.0);
  std::fill(Qx.begin(), Qx.begin() + (T - 1) * dim_state_derivative, 0.0);
  std::fill(Qu.begin(), Qu.begin() + (T - 1) * dim_action, 0.0);
  mju_zero(dV, 2);
}

// compute gradient at time step
int Gradient::GradientStep(int n, int m, const double *Wx, const double *At,
                           const double *Bt, const double *cxt,
                           const double *cut, double *Vxt, double *dut,
                           double *Qxt, double *Qut) {
  //    Qx = cx + A'*Wx
  mju_mulMatTVec(Qxt, At, Wx, n, n);
  mju_addTo(Qxt, cxt, n);

  //    Qu  = cu + B'*Wx;
  mju_mulMatTVec(Qut, Bt, Wx, n, m);
  mju_addTo(Qut, cut, m);

  //    k = -Qu
  mju_scl(dut, Qut, -1.0, m);

  // update cost-to-go
  mju_copy(Vxt, Qxt, n);

  // dV = dV + [du'*Qu]
  dV[0] += mju_dot(dut, Qut, m);

  return 1;
}

// compute gradient for entire trajectory
int Gradient::Compute(GradientPolicy *p, const ModelDerivatives *md,
                      const CostDerivatives *cd, int dim_state_derivative,
                      int dim_action, int T) {
  // reset
  mju_zero(dV, 2);

  // final DerivativeStep cost-to-go
  mju_copy(DataAt(Vx, (T - 1) * dim_state_derivative),
           DataAt(cd->cx, (T - 1) * dim_state_derivative),
           dim_state_derivative);

  // // iterate gradient steps backward in time
  int time_index = T - 1;
  for (int t = T - 1; t > 0; t--) {
    int status = this->GradientStep(
        dim_state_derivative, dim_action, DataAt(Vx, t * dim_state_derivative),
        DataAt(md->A, (t - 1) * dim_state_derivative * dim_state_derivative),
        DataAt(md->B, (t - 1) * dim_state_derivative * dim_action),
        DataAt(cd->cx, (t - 1) * dim_state_derivative),
        DataAt(cd->cu, (t - 1) * dim_action),
        DataAt(Vx, (t - 1) * dim_state_derivative),
        DataAt(p->k, (t - 1) * dim_action),
        DataAt(Qx, (t - 1) * dim_state_derivative),
        DataAt(Qu, (t - 1) * dim_action));

    // failure
    if (!status) {
      time_index = t - 1;
      break;
    }

    // complete
    if (t == 1) {
      mju_copy(DataAt(p->k, (T - 1) * dim_action),
               DataAt(p->k, (T - 2) * dim_action), dim_action);
      return 0;
    }
  }

  return time_index;
}

}  // namespace mjpc
