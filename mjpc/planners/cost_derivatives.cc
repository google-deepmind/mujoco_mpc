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

#include "mjpc/planners/cost_derivatives.h"

#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/norm.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void CostDerivatives::Allocate(int dim_state_derivative, int dim_action,
                               int dim_residual, int T, int dim_max) {
  // norm derivatives
  cr.resize(dim_residual * T);
  crr.resize(dim_residual * dim_residual * T);

  // cost gradients
  cx.resize(dim_state_derivative * T);
  cu.resize(dim_action * T);

  // cost Hessians
  cxx.resize(dim_state_derivative * dim_state_derivative * T);
  cuu.resize(dim_action * dim_action * T);
  cxu.resize(dim_state_derivative * dim_action * T);

  // scratch space
  c_scratch_.resize(T * dim_max * dim_max);
  cx_scratch_.resize(T * dim_state_derivative);
  cu_scratch_.resize(T * dim_action);
  cxx_scratch_.resize(T * dim_state_derivative * dim_state_derivative);
  cuu_scratch_.resize(T * dim_action * dim_action);
  cxu_scratch_.resize(T * dim_state_derivative * dim_action);
}

// reset memory to zeros
void CostDerivatives::Reset(int dim_state_derivative, int dim_action,
                            int dim_residual, int T) {
  std::fill(cr.begin(), cr.begin() + dim_residual * T, 0.0);
  std::fill(cx.begin(), cx.begin() + dim_state_derivative * T, 0.0);
  std::fill(cu.begin(), cu.begin() + dim_action * T, 0.0);
  std::fill(crr.begin(), crr.begin() + dim_residual * dim_residual * T, 0.0);
  std::fill(cxx.begin(),
            cxx.begin() + dim_state_derivative * dim_state_derivative * T, 0.0);
  std::fill(cuu.begin(), cuu.begin() + dim_action * dim_action * T, 0.0);
  std::fill(cxu.begin(), cxu.begin() + dim_state_derivative * dim_action * T,
            0.0);
  std::fill(cx_scratch_.begin(), cx_scratch_.begin() + T * dim_state_derivative,
            0.0);
  std::fill(cu_scratch_.begin(), cu_scratch_.begin() + T * dim_action, 0.0);
  std::fill(
      cxx_scratch_.begin(),
      cxx_scratch_.begin() + T * dim_state_derivative * dim_state_derivative,
      0.0);
  std::fill(cuu_scratch_.begin(),
            cuu_scratch_.begin() + T * dim_action * dim_action, 0.0);
  std::fill(cxu_scratch_.begin(),
            cxu_scratch_.begin() + T * dim_state_derivative * dim_action, 0.0);
}

// compute derivatives at one time step
double CostDerivatives::DerivativeStep(
    double* Cx, double* Cu, double* Cxx, double* Cuu, double* Cxu, double* Cr,
    double* Crr, double* C_scratch, double* Cx_scratch, double* Cu_scratch,
    double* Cxx_scratch, double* Cuu_scratch, double* Cxu_scratch,
    const double* r, const double* rx, const double* ru, int nr, int nx,
    int dim_action, double weight, const double* p, NormType type) {
  // norm derivatives
  double C = Norm(Cr, Crr, r, p, nr, type);

  // cx
  mju_mulMatTVec(Cx_scratch, rx, Cr, nr, nx);
  mju_addToScl(Cx, Cx_scratch, weight, nx);

  // cu
  mju_mulMatTVec(Cu_scratch, ru, Cr, nr, dim_action);
  mju_addToScl(Cu, Cu_scratch, weight, dim_action);

  // cxx
  mju_mulMatMat(C_scratch, Crr, rx, nr, nr, nx);
  mju_mulMatTMat(Cxx_scratch, C_scratch, rx, nr, nx, nx);
  mju_addToScl(Cxx, Cxx_scratch, weight, nx * nx);

  // cxu
  mju_mulMatTMat(Cxu_scratch, C_scratch, ru, nr, nx, dim_action);
  mju_addToScl(Cxu, Cxu_scratch, weight, nx * dim_action);

  // cuu
  mju_mulMatMat(C_scratch, Crr, ru, nr, nr, dim_action);
  mju_mulMatTMat(Cuu_scratch, C_scratch, ru, nr, dim_action, dim_action);
  mju_addToScl(Cuu, Cuu_scratch, weight, dim_action * dim_action);

  return weight * C;
}

// compute derivatives at all time steps
void CostDerivatives::Compute(double* r, double* rx, double* ru,
                              int dim_state_derivative, int dim_action,
                              int dim_max, int num_sensors, int num_residual,
                              const int* dim_norm_residual, int num_term,
                              const double* weights, const NormType* norms,
                              const double* parameters,
                              const int* num_norm_parameter, double risk,
                              int T, ThreadPool& pool) {
  // reset
  this->Reset(dim_state_derivative, dim_action, num_residual, T);
  {
    int count_before = pool.GetCount();
    for (int t = 0; t < T; t++) {
      pool.Schedule([&cd = *this, &r, &rx, &ru, num_term, num_residual,
                     &dim_norm_residual, &weights, &norms, &parameters,
                     &num_norm_parameter, risk, num_sensors,
                     dim_state_derivative, dim_action, dim_max, t, T]() {
        // ----- term derivatives ----- //
        int f_shift = 0;
        int p_shift = 0;
        double c = 0.0;
        for (int i = 0; i < num_term; i++) {
          c += cd.DerivativeStep(
              DataAt(cd.cx, t * dim_state_derivative),
              DataAt(cd.cu, t * dim_action),
              DataAt(cd.cxx, t * dim_state_derivative * dim_state_derivative),
              DataAt(cd.cuu, t * dim_action * dim_action),
              DataAt(cd.cxu, t * dim_state_derivative * dim_action),
              DataAt(cd.cr, t * num_residual),
              DataAt(cd.crr, t * num_residual * num_residual),
              DataAt(cd.c_scratch_, t * dim_max * dim_max),
              DataAt(cd.cx_scratch_, t * dim_state_derivative),
              DataAt(cd.cu_scratch_, t * dim_action),
              DataAt(cd.cxx_scratch_,
                     t * dim_state_derivative * dim_state_derivative),
              DataAt(cd.cuu_scratch_, t * dim_action * dim_action),
              DataAt(cd.cxu_scratch_, t * dim_state_derivative * dim_action),
              r + t * num_residual + f_shift,
              rx + t * num_sensors * dim_state_derivative +
                  f_shift * dim_state_derivative,
              ru + t * num_sensors * dim_action + f_shift * dim_action,
              dim_norm_residual[i], dim_state_derivative, dim_action,
              weights[i] / T, parameters + p_shift, norms[i]);

          f_shift += dim_norm_residual[i];
          p_shift += num_norm_parameter[i];
        }

        // ----- risk transformation ----- //
        if (mju_abs(risk) < kRiskNeutralTolerance) {
          return;
        }

        double s = mju_exp(risk * c);

        // cx
        mju_scl(DataAt(cd.cx, t * dim_state_derivative),
                DataAt(cd.cx, t * dim_state_derivative), s,
                dim_state_derivative);

        // cu
        mju_scl(DataAt(cd.cu, t * dim_action), DataAt(cd.cu, t * dim_action), s,
                dim_action);

        // cxx
        mju_scl(DataAt(cd.cxx, t * dim_state_derivative * dim_state_derivative),
                DataAt(cd.cxx, t * dim_state_derivative * dim_state_derivative),
                s, dim_state_derivative * dim_state_derivative);
        mju_mulMatMat(DataAt(cd.cxx_scratch_,
                             t * dim_state_derivative * dim_state_derivative),
                      DataAt(cd.cx, t * dim_state_derivative),
                      DataAt(cd.cx, t * dim_state_derivative),
                      dim_state_derivative, 1, dim_state_derivative);
        mju_scl(DataAt(cd.cxx_scratch_,
                       t * dim_state_derivative * dim_state_derivative),
                DataAt(cd.cxx_scratch_,
                       t * dim_state_derivative * dim_state_derivative),
                risk * s, dim_state_derivative * dim_state_derivative);
        mju_addTo(
            DataAt(cd.cxx, t * dim_state_derivative * dim_state_derivative),
            DataAt(cd.cxx_scratch_,
                   t * dim_state_derivative * dim_state_derivative),
            dim_state_derivative * dim_state_derivative);

        // cxu
        mju_scl(DataAt(cd.cxu, t * dim_state_derivative * dim_action),
                DataAt(cd.cxu, t * dim_state_derivative * dim_action), s,
                dim_state_derivative * dim_action);
        mju_mulMatMat(
            DataAt(cd.cxu_scratch_, t * dim_state_derivative * dim_action),
            DataAt(cd.cx, t * dim_state_derivative),
            DataAt(cd.cu, t * dim_action), dim_state_derivative, 1, dim_action);
        mju_scl(DataAt(cd.cxu_scratch_, t * dim_state_derivative * dim_action),
                DataAt(cd.cxu_scratch_, t * dim_state_derivative * dim_action),
                risk * s, dim_state_derivative * dim_action);
        mju_addTo(
            DataAt(cd.cxu, t * dim_state_derivative * dim_action),
            DataAt(cd.cxu_scratch_, t * dim_state_derivative * dim_action),
            dim_state_derivative * dim_action);

        // cuu
        mju_scl(DataAt(cd.cuu, t * dim_action * dim_action),
                DataAt(cd.cuu, t * dim_action * dim_action), s,
                dim_action * dim_action);
        mju_mulMatMat(DataAt(cd.cuu_scratch_, t * dim_action * dim_action),
                      DataAt(cd.cu, t * dim_action),
                      DataAt(cd.cu, t * dim_action), dim_action, 1, dim_action);
        mju_scl(DataAt(cd.cuu_scratch_, t * dim_action * dim_action),
                DataAt(cd.cuu_scratch_, t * dim_action * dim_action), risk * s,
                dim_action * dim_action);
        mju_addTo(DataAt(cd.cuu, t * dim_action * dim_action),
                  DataAt(cd.cuu_scratch_, t * dim_action * dim_action),
                  dim_action * dim_action);
      });
    }
    pool.WaitCount(count_before + T);
  }
  pool.ResetCount();
}

}  // namespace mjpc
