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

#include "mjpc/planners/ilqg/backward_pass.h"

#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/planners/cost_derivatives.h"
#include "mjpc/planners/ilqg/boxqp.h"
#include "mjpc/planners/ilqg/policy.h"
#include "mjpc/planners/ilqg/settings.h"
#include "mjpc/planners/model_derivatives.h"
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void iLQGBackwardPass::Allocate(int dim_dstate, int dim_action, int T) {
  Vx.resize(dim_dstate * T);
  Vxx.resize(dim_dstate * dim_dstate * T);
  Qx.resize(dim_dstate * (T - 1));
  Qu.resize(dim_action * (T - 1));
  Qxx.resize(dim_dstate * dim_dstate * (T - 1));
  Qxu.resize(dim_dstate * dim_action * (T - 1));
  Quu.resize(dim_action * dim_action * (T - 1));
  Q_scratch.resize(
      10 *
      (dim_dstate * dim_dstate + 7 * dim_action + 2 * dim_action * dim_action + dim_dstate * dim_action +
       3 * mju_max(dim_action, dim_dstate) * mju_max(dim_action, dim_dstate)));

  // regularization
  regularization = 1.0;
  regularization_rate = 1.0;
  regularization_factor = 2.0;
}

// reset memory to zeros
void iLQGBackwardPass::Reset(int dim_dstate, int dim_action, int T) {
  mju_zero(dV, 2);
  std::fill(Vx.begin(), Vx.end(), 0.0);
  std::fill(Vxx.begin(), Vxx.end(), 0.0);
  std::fill(Qx.begin(), Qx.end(), 0.0);
  std::fill(Qu.begin(), Qu.end(), 0.0);
  std::fill(Qxx.begin(), Qxx.end(), 0.0);
  std::fill(Qxu.begin(), Qxu.end(), 0.0);
  std::fill(Quu.begin(), Quu.end(), 0.0);
  regularization = 1.0;
  regularization_rate = 1.0;
  regularization_factor = 2.0;
}

// Riccati at one time step
int iLQGBackwardPass::RiccatiStep(
    int n, int m, double mu, const double *Wx, const double *Wxx,
    const double *At, const double *Bt, const double *cxt, const double *cut,
    const double *cxxt, const double *cxut, const double *cuut, double *Vxt,
    double *Vxxt, double *dut, double *Kt, double *dV, double *Qxt, double *Qut,
    double *Qxxt, double *Qxut, double *Quut, double *scratch, BoxQP &boxqp,
    const double *action, const double *action_limits, int reg_type,
    int limits) {
  int i, mmn = mju_max(m, n);
  mjtNum *Vxx_reg, *Quu_reg, *Qxu_reg, *tmp, *tmp2, *tmp3;

  // allocate workspace variables
  Vxx_reg = scratch;
  scratch += n * n;
  Qxu_reg = scratch;
  scratch += n * m;
  Quu_reg = scratch;
  scratch += m * m;
  tmp = scratch;
  scratch += mmn * mmn;
  tmp2 = scratch;
  scratch += mmn * mmn;
  tmp3 = scratch;
  scratch += mmn * mmn;

  //----- compute Qut,Qxut,Quut,Qxt,Qxxt ----- //
  //    tmp  = At'*Wxx
  mju_mulMatTMat(tmp, At, Wxx, n, n, n);

  //    Qxt = cxt + At'*Wx
  mju_mulMatTVec(Qxt, At, Wx, n, n);
  mju_addTo(Qxt, cxt, n);

  //    Qxxt = cxxt + tmp*At
  mju_mulMatMat(Qxxt, tmp, At, n, n, n);
  mju_addTo(Qxxt, cxxt, n * n);

  //    Qut  = cut + Bt'*Wx;
  mju_mulMatTVec(Qut, Bt, Wx, n, m);
  mju_addTo(Qut, cut, m);

  //    Qxut = cxut  + tmp*Bt;
  mju_mulMatMat(Qxut, tmp, Bt, n, n, m);
  mju_addTo(Qxut, cxut, n * m);

  //    Quut = cuut + Bt'*Wxx*Bt;
  mju_mulMatTMat(tmp2, Bt, Wxx, n, m, n);
  mju_mulMatMat(Quut, tmp2, Bt, m, n, m);
  mju_addTo(Quut, cuut, m * m);

  //----- regularize ----- //
  if (reg_type == kValueRegularization) {
    // regularize cost-to-go
    mju_copy(Vxx_reg, Wxx, n * n);
    for (int i = 0; i < n; i++) {
      Vxx_reg[n * i + i] += mu;
    }

    // Qxu_reg = cxut + At'*Vxx_reg*Bt
    //    tmp  = At'*Vxx_reg
    mju_mulMatTMat(tmp, At, Vxx_reg, n, n, n);
    //    Qxu_reg = cxut  + tmp*Bt;
    mju_mulMatMat(Qxu_reg, tmp, Bt, n, n, m);
    mju_addTo(Qxu_reg, cxut, n * m);

    // Quu_reg = cuut + Bt'*Vxx_reg*Bt;
    //    tmp = Bt'*Vxx_reg
    mju_mulMatTMat(tmp2, Bt, Vxx_reg, n, m, n);
    mju_mulMatMat(Quu_reg, tmp2, Bt, m, n, m);
    mju_addTo(Quu_reg, cuut, m * m);
  } else {
    mju_copy(Qxu_reg, Qxut, n * m);
    mju_copy(Quu_reg, Quut, m * m);
  }

  if (mu) {
    if (reg_type == kControlRegularization) {
      for (i = 0; i < m; i++) {
        Quu_reg[i * m + i] += mu;  // Quu_reg = Quut + mu*eye(m)
      }
    } else if (reg_type == kStateControlRegularization) {
      mju_mulMatTMat(tmp, At, Bt, n, n, m);
      mju_addToScl(Qxu_reg, tmp, mu,
                   m * n);  // Qxu_reg = Qxut + mu*At'*Bt
      mju_mulMatTMat(tmp, Bt, Bt, n, m, m);
      mju_addToScl(Quu_reg, tmp, mu,
                   m * m);  // Quu_reg = Quut + mu*Bt'*Bt
    }
  }

  // feedback gains Kt
  mju_zero(Kt, n * m);

  if (limits == 1) {
    // set problem
    mju_scl(boxqp.H.data(), Quu_reg, 1.0, m * m);
    mju_scl(boxqp.g.data(), Qut, 1.0, m);

    for (int i = 0; i < m; i++) {
      boxqp.lower[i] = action_limits[2 * i] - action[i];
      boxqp.upper[i] = action_limits[2 * i + 1] - action[i];
    }

    // solve constrained quadratic program
    int mFree = mju_boxQP(boxqp.res.data(), boxqp.R.data(), boxqp.index.data(),
                          boxqp.H.data(), boxqp.g.data(), m, boxqp.lower.data(),
                          boxqp.upper.data());
    if (mFree < 0) {
      return 0;
    }

    // tmp = compress_free(Qxut)
    for (int i = 0; i < mFree; i++) {
      for (int j = 0; j < n; j++) {
        tmp[mFree * j + i] = Qxut[m * j + boxqp.index[i]];
      }
    }

    // tmp2 = H_free\Qxu_free
    for (int i = 0; i < n; i++) {
      mju_cholSolve(tmp2 + i * mFree, boxqp.R.data(), tmp + i * mFree, mFree);
    }

    // Kt = expand_free(-tmp2)'
    for (int i = 0; i < mFree; i++) {
      for (int j = 0; j < n; j++) {
        Kt[j + n * boxqp.index[i]] = -tmp2[j * mFree + i];
      }
    }

    // dut - solution to QP
    mju_copy(dut, boxqp.res.data(), m);
  } else {
    // Quut^-1
    mju_copy(tmp3, Quu_reg, m * m);
    int rank = mju_cholFactor(tmp3, m, 0.0);

    if (rank < m) {
      printf("backward pass failure\n");
      return 0;
    }

    // Kt = - Quut \ Qxut
    for (int i = 0; i < n; i++) {
      mju_cholSolve(tmp + i * m, tmp3, Qxut + i * m, m);
    }
    mju_transpose(Kt, tmp, n, m);
    mju_scl(Kt, Kt, -1.0, m * n);

    // dut = - Quut \ Qut
    mju_cholSolve(dut, tmp3, Qut, m);
    mju_scl(dut, dut, -1.0, m);
  }
  //----- update cost-to-go ----- //
  // copy uncontrolled derivatives
  mju_copy(Vxt, Qxt, n);
  mju_copy(Vxxt, Qxxt, n * n);

  // dV = dV + [dut'*Qut  .5*dut'*Quut*dut]
  dV[0] += mju_dot(dut, Qut, m);
  mju_mulMatVec(tmp, Quut, dut, m, m);
  dV[1] += 0.5 * mju_dot(dut, tmp, m);

  // Vxt += Kt'*(Quut*dut + Qut) + Qxut*dut
  mju_add(tmp2, tmp, Qut, m);  // reuse tmp=Quut*dut
  mju_mulMatTVec(tmp, Kt, tmp2, m, n);
  mju_addTo(Vxt, tmp, n);
  mju_mulMatVec(tmp, Qxut, dut, n, m);
  mju_addTo(Vxt, tmp, n);

  // tmp3 = Kt'*Quut*Kt
  mju_mulMatMat(Qxu_reg, Quut, Kt, m, m, n);  // use Qxu_reg as tmp
  mju_mulMatTMat(tmp3, Kt, Qxu_reg, m, n, n);
  mju_addTo(Vxxt, tmp3, n * n);

  // tmp = Qxut*Kt + Kt'*Qxut'
  mju_mulMatMat(tmp2, Qxut, Kt, n, m, n);
  mju_copy(tmp, tmp2, n * n);
  mju_transpose(tmp3, tmp2, n, n);
  mju_addTo(tmp, tmp3, n * n);

  // Vxxt += Kt'*Quut*Kt + Qxut*Kt + Kt'*Qxut'
  mju_addTo(Vxxt, tmp, n * n);
  mju_symmetrize(Vxxt, Vxxt, n);
  return 1;
}

// compute backward pass using Riccati
int iLQGBackwardPass::Riccati(iLQGPolicy *p, const ModelDerivatives *md,
                          const CostDerivatives *cd, int dim_dstate,
                          int dim_action, int T, double reg, BoxQP &boxqp,
                          const double *actions, const double *action_limits,
                          const iLQGSettings &settings) {
  // reset
  mju_zero(dV, 2);

  // final DerivativeStep cost-to-go
  mju_copy(DataAt(Vx, (T - 1) * dim_dstate),
           DataAt(cd->cx, (T - 1) * dim_dstate), dim_dstate);
  mju_copy(DataAt(Vxx, (T - 1) * dim_dstate * dim_dstate),
           DataAt(cd->cxx, (T - 1) * dim_dstate * dim_dstate),
           dim_dstate * dim_dstate);

  // iterate Riccati backward in time
  int bp_iter = 0;
  int time_index = T - 1;
  while (bp_iter < settings.max_regularization_iterations) {
    for (int t = T - 1; t > 0; t--) {
      int status = this->RiccatiStep(
          dim_dstate, dim_action, reg, DataAt(Vx, t * dim_dstate),
          DataAt(Vxx, t * dim_dstate * dim_dstate),
          DataAt(md->A, (t - 1) * dim_dstate * dim_dstate),
          DataAt(md->B, (t - 1) * dim_dstate * dim_action),
          DataAt(cd->cx, (t - 1) * dim_dstate),
          DataAt(cd->cu, (t - 1) * dim_action),
          DataAt(cd->cxx, (t - 1) * dim_dstate * dim_dstate),
          DataAt(cd->cxu, (t - 1) * dim_dstate * dim_action),
          DataAt(cd->cuu, (t - 1) * dim_action * dim_action),
          DataAt(Vx, (t - 1) * dim_dstate),
          DataAt(Vxx, (t - 1) * dim_dstate * dim_dstate),
          DataAt(p->action_improvement, (t - 1) * dim_action),
          DataAt(p->feedback_gain, (t - 1) * dim_action * dim_dstate), dV,
          DataAt(Qx, (t - 1) * dim_dstate), DataAt(Qu, (t - 1) * dim_action),
          DataAt(Qxx, (t - 1) * dim_dstate * dim_dstate),
          DataAt(Qxu, (t - 1) * dim_dstate * dim_action),
          DataAt(Quu, (t - 1) * dim_action * dim_action), Q_scratch.data(),
          boxqp, actions + (t - 1) * dim_action, action_limits,
          settings.regularization_type, settings.action_limits);

      // failure
      if (!status) {
        time_index = t - 1;
        break;
      }

      // complete
      if (t == 1) {
        mju_copy(DataAt(p->feedback_gain, (T - 1) * dim_action * dim_dstate),
                 DataAt(p->feedback_gain, (T - 2) * dim_action * dim_dstate),
                 dim_action * dim_dstate);
        mju_copy(DataAt(p->action_improvement, (T - 1) * dim_action),
                 DataAt(p->action_improvement, (T - 2) * dim_action),
                 dim_action);
        return 0;
      }
    }

    // increase regularization
    if (regularization <= settings.max_regularization) {
      this->ScaleRegularization(regularization_factor,
                                settings.min_regularization,
                                settings.max_regularization);
      bp_iter += 1;
    } else {
      return time_index;
    }
  }
  return time_index;
}

// scale backward pass regularization
void iLQGBackwardPass::ScaleRegularization(double factor, double reg_min,
                                       double reg_max) {
  // scale rate
  if (factor > 1)
    regularization_rate = mju_max(regularization_rate * factor, factor);
  else
    regularization_rate = mju_min(regularization_rate * factor, factor);

  // scale regularization
  regularization =
      mju_min(mju_max(regularization * regularization_rate, reg_min), reg_max);
}

// update backward pass regularization
void iLQGBackwardPass::UpdateRegularization(double reg_min, double reg_max,
                                        double z, double s) {
  // divergence or no improvement: increase dmu by muFactor^2
  if (mju_isBad(z) || mju_isBad(s))
    // muScale(t, o, o.muFactor*o.muFactor);
    this->ScaleRegularization(regularization_factor * regularization_factor,
                              reg_min, reg_max);
  // sufficient improvement: decrease dmu by muFactor
  else if (z > 0.5 || s > 0.3)
    // muScale(t, o, 1/o.muFactor);
    this->ScaleRegularization(1.0 / regularization_factor, reg_min, reg_max);
  // insufficient improvement: increase dmu by muFactor
  else if (z < 0.1 || s < 0.06)
    // muScale(t, o, o.muFactor);
    this->ScaleRegularization(regularization_factor, reg_min, reg_max);
}

}  // namespace mjpc
