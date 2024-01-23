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
                               ThreadPool& pool) {
  int count_before = pool.GetCount();

  // t == 0
  pool.Schedule([&m, &data, &A = A, &B = B, &C = C, &D = D, &x, &u, &h,
                  dim_state, dim_state_derivative, dim_action, dim_sensor, tol,
                  mode]() {
    int t = 0;
    mjData* d = data[ThreadPool::WorkerId()].get();
    // set state
    SetState(m, d, x + t * dim_state);
    d->time = h[t];

    // set action
    mju_copy(d->ctrl, u + t * dim_action, dim_action);

    // derivatives
    mjd_transitionFD(
        m, d, tol, mode,
        DataAt(A, t * (dim_state_derivative * dim_state_derivative)),
        DataAt(B, t * (dim_state_derivative * dim_action)),
        DataAt(C, t * (dim_sensor * dim_state_derivative)),
        DataAt(D, t * (dim_sensor * dim_action)));
  });

  // t == T - 2
  pool.Schedule([&m, &data, &A = A, &B = B, &C = C, &D = D, &x, &u, &h,
                  dim_state, dim_state_derivative, dim_action, dim_sensor, tol,
                  mode, T]() {
    int t = T - 2;
    mjData* d = data[ThreadPool::WorkerId()].get();
    // set state
    SetState(m, d, x + t * dim_state);
    d->time = h[t];

    // set action
    mju_copy(d->ctrl, u + t * dim_action, dim_action);

    // derivatives
    mjd_transitionFD(
        m, d, tol, mode,
        DataAt(A, t * (dim_state_derivative * dim_state_derivative)),
        DataAt(B, t * (dim_state_derivative * dim_action)),
        DataAt(C, t * (dim_sensor * dim_state_derivative)),
        DataAt(D, t * (dim_sensor * dim_action)));
  });

  // t == T - 1
  pool.Schedule([&m, &data, &C = C, &x, &u, &h,
                  dim_state, dim_state_derivative, dim_action, dim_sensor, tol,
                  mode, T]() {
    int t = T - 1;
    mjData* d = data[ThreadPool::WorkerId()].get();
    
    // set state
    SetState(m, d, x + t * dim_state);
    d->time = h[t];

    // set action
    mju_copy(d->ctrl, u + t * dim_action, dim_action);

    // Jacobians
    mjd_transitionFD(m, d, tol, mode, nullptr, nullptr,
                      DataAt(C, t * (dim_sensor * dim_state_derivative)),
                      nullptr);
  });

  // skip values
  std::vector<int> tt;
  tt.push_back(0);
  
  int skip = 3;
  for (int t = skip; t < T - skip; t += skip) {
    tt.push_back(t);
  }
  tt.push_back(T - 2);
  tt.push_back(T - 1);
  int S = tt.size();

  for (int t = skip; t < T - skip; t += skip) {
    pool.Schedule([&m, &data, &A = A, &B = B, &C = C, &D = D, &x, &u, &h,
                    dim_state, dim_state_derivative, dim_action, dim_sensor,
                    tol, mode, t, T]() {
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
  pool.WaitCount(count_before + S);
  pool.ResetCount();

  // -- interpolate skipped values -- //

  // find skipped indices
  std::vector<int> ss;
  for (int t = 0; t < T; t++) {
    if(std::find(tt.begin(), tt.end(), t) == tt.end()) {
      /* v contains x */
      ss.push_back(t);
    } 
  }

  // convert to double index for FindInterval
  std::vector<double> ttd;
  for (int i: tt) {
    ttd.push_back((double)i);
  }

  int H = ss.size();
  for (int i: ss) {
    pool.Schedule([&A = A, &B = B, &C = C, &D = D, &tt, &ttd,
                   dim_state_derivative, dim_action, dim_sensor, i]() {
      // find interval
      int bounds[2];
      FindInterval(bounds, ttd, (double)i, ttd.size());

      // normalized time
      double q = double(i - tt[bounds[0]]) / double(tt[bounds[1]] - tt[bounds[0]]);
      if (bounds[0] == bounds[1]) {
        q = 0.0;
      }

      // A
      int nA = dim_state_derivative * dim_state_derivative;
      double* Ai = DataAt(A, i * nA);
      double* AL = DataAt(A, tt[bounds[0]] * nA);
      double* AU = DataAt(A, tt[bounds[1]] * nA);

      mju_scl(Ai, AL, 1.0 - q, nA);
      mju_addToScl(Ai, AU, q, nA);
            
      // B 
      int nB = dim_state_derivative * dim_action;
      double* Bi = DataAt(B, i * nB);
      double* BL = DataAt(B, tt[bounds[0]] * nB);
      double* BU = DataAt(B, tt[bounds[1]] * nB);

      mju_scl(Bi, BL, 1.0 - q, nB);
      mju_addToScl(Bi, BU, q, nB);

      // C
      int nC = dim_sensor * dim_state_derivative;
      double* Ci = DataAt(C, i * nC);
      double* CL = DataAt(C, tt[bounds[0]] * nC);
      double* CU = DataAt(C, tt[bounds[1]] * nC);

      mju_scl(Ci, CL, 1.0 - q, nC);
      mju_addToScl(Ci, CU, q, nC);

      // D
      int nD = dim_sensor * dim_action;
      double* Di = DataAt(D, i * nD);
      double* DL = DataAt(D, tt[bounds[0]] * nD);
      double* DU = DataAt(D, tt[bounds[1]] * nD);

      mju_scl(Di, DL, 1.0 - q, nD);
      mju_addToScl(Di, DU, q, nD);
    });
  }

  pool.WaitCount(count_before + H);
  pool.ResetCount();
}

}  // namespace mjpc
