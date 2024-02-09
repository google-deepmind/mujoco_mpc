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

#include "mjpc/planners/model_derivatives.h"

#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void ModelDerivatives::Allocate(int dim_state_derivative, int dim_action,
                                int dim_sensor, int T) {
  A.resize(dim_state_derivative * dim_state_derivative * T);
  B.resize(dim_state_derivative * dim_action * T);
  C.resize(dim_sensor * dim_state_derivative * T);
  D.resize(dim_sensor * dim_action * T);
}

// reset memory to zeros
void ModelDerivatives::Reset(int dim_state_derivative, int dim_action,
                             int dim_sensor, int T) {
  std::fill(A.begin(),
            A.begin() + T * dim_state_derivative * dim_state_derivative, 0.0);
  std::fill(B.begin(), B.begin() + T * dim_state_derivative * dim_action, 0.0);
  std::fill(C.begin(), C.begin() + T * dim_sensor * dim_state_derivative, 0.0);
  std::fill(D.begin(), D.begin() + T * dim_sensor * dim_action, 0.0);
}

// compute derivatives at all time steps
void ModelDerivatives::Compute(const mjModel* m,
                               const std::vector<UniqueMjData>& data,
                               const double* x, const double* u,
                               const double* h, int dim_state,
                               int dim_state_derivative, int dim_action,
                               int dim_sensor, int T, double tol, int mode,
                               ThreadPool& pool, int skip) {
  // reset indices
  evaluate_.clear();
  interpolate_.clear();

  // evaluate indices
  int s = skip + 1;
  evaluate_.push_back(0);
  for (int t = s; t < T - s; t += s) {
    evaluate_.push_back(t);
  }
  evaluate_.push_back(T - 2);
  evaluate_.push_back(T - 1);

  // interpolate indices
  for (int t = 0, e = 0; t < T; t++) {
    if (e == evaluate_.size() || evaluate_[e] > t) {
      interpolate_.push_back(t);
    } else {
      e++;
    }
  }

  // evaluate derivatives
  int count_before = pool.GetCount();
  for (int t : evaluate_) {
    pool.Schedule([&m, &data, &A = A, &B = B, &C = C, &D = D, &x, &u, &h,
                   dim_state, dim_state_derivative, dim_action, dim_sensor, tol,
                   mode, t, T]() {
      mjData* d = data[ThreadPool::WorkerId()].get();
      // set state
      SetState(m, d, x + t * dim_state);
      d->time = h[t];

      // set action
      mju_copy(d->ctrl, u + t * dim_action, dim_action);

      // Jacobians
      if (t == T - 1) {
        // Jacobians
        mjd_transitionFD(m, d, tol, mode, nullptr, nullptr,
                         DataAt(C, t * (dim_sensor * dim_state_derivative)),
                         nullptr);
      } else {
        // derivatives
        mjd_transitionFD(
            m, d, tol, mode,
            DataAt(A, t * (dim_state_derivative * dim_state_derivative)),
            DataAt(B, t * (dim_state_derivative * dim_action)),
            DataAt(C, t * (dim_sensor * dim_state_derivative)),
            DataAt(D, t * (dim_sensor * dim_action)));
      }
    });
  }
  pool.WaitCount(count_before + evaluate_.size());
  pool.ResetCount();

  // interpolate derivatives
  count_before = pool.GetCount();
  for (int t : interpolate_) {
    pool.Schedule([&A = A, &B = B, &C = C, &D = D, &evaluate_ = this->evaluate_,
                   dim_state_derivative, dim_action, dim_sensor, t]() {
      // find interval
      int bounds[2];
      FindInterval(bounds, evaluate_, t, evaluate_.size());
      int e0 = evaluate_[bounds[0]];
      int e1 = evaluate_[bounds[1]];

      // normalized input
      double tt = double(t - e0) / double(e1 - e0);
      if (bounds[0] == bounds[1]) {
        tt = 0.0;
      }

      // A
      int nA = dim_state_derivative * dim_state_derivative;
      double* Ai = DataAt(A, t * nA);
      const double* AL = DataAt(A, e0 * nA);
      const double* AU = DataAt(A, e1 * nA);

      mju_scl(Ai, AL, 1.0 - tt, nA);
      mju_addToScl(Ai, AU, tt, nA);

      // B
      int nB = dim_state_derivative * dim_action;
      double* Bi = DataAt(B, t * nB);
      const double* BL = DataAt(B, e0 * nB);
      const double* BU = DataAt(B, e1 * nB);

      mju_scl(Bi, BL, 1.0 - tt, nB);
      mju_addToScl(Bi, BU, tt, nB);

      // C
      int nC = dim_sensor * dim_state_derivative;
      double* Ci = DataAt(C, t * nC);
      const double* CL = DataAt(C, e0 * nC);
      const double* CU = DataAt(C, e1 * nC);

      mju_scl(Ci, CL, 1.0 - tt, nC);
      mju_addToScl(Ci, CU, tt, nC);

      // D
      int nD = dim_sensor * dim_action;
      double* Di = DataAt(D, t * nD);
      const double* DL = DataAt(D, e0 * nD);
      const double* DU = DataAt(D, e1 * nD);

      mju_scl(Di, DL, 1.0 - tt, nD);
      mju_addToScl(Di, DU, tt, nD);
    });
  }

  pool.WaitCount(count_before + interpolate_.size());
  pool.ResetCount();
}

}  // namespace mjpc
