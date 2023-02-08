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

#ifndef MJPC_PLANNERS_COST_DERIVATIVES_H_
#define MJPC_PLANNERS_COST_DERIVATIVES_H_

#include <vector>

#include "mjpc/norm.h"
#include "mjpc/threadpool.h"

namespace mjpc {

// data and methods for cost derivatives
class CostDerivatives {
 public:
  // constructor
  CostDerivatives() = default;

  // destructor
  ~CostDerivatives() = default;

  // ----- methods ----- //

  // allocate memory
  void Allocate(int dim_state_derivative, int dim_action, int dim_residual,
                int T, int dim_max);

  // reset memory to zeros
  void Reset(int dim_state_derivative, int dim_action, int dim_residual, int T);

  // compute derivatives at one time step
  double DerivativeStep(double* Cx, double* Cu, double* Cxx, double* Cuu,
                        double* Cxu, double* Cr, double* Crr, double* C_scratch,
                        double* Cx_scratch, double* Cu_scratch,
                        double* Cxx_scratch, double* Cuu_scratch,
                        double* Cxu_scratch, const double* r, const double* rx,
                        const double* ru, int nr, int nx, int dim_action,
                        double weight, const double* p, NormType type);

  // compute derivatives at all time steps
  void Compute(double* r, double* rx, double* ru, int dim_state_derivative,
               int dim_action, int dim_max, int num_sensors, int num_residual,
               const int* dim_norm_residual, int num_term,
               const double* weights, const NormType* norms,
               const double* parameters, const int* num_norm_parameter,
               double risk, int T, ThreadPool& pool);

  std::vector<double> cr;   // norm gradient wrt residual
                            //   (T * dim_residual)
  std::vector<double> crr;  // norm Hessian wrt residual
                            //   (T * dim_residual * dim_residual)
  std::vector<double> cx;   // cost gradient wrt state
                            //   (T * dim_state_derivative)
  std::vector<double> cu;   // cost gradient wrt action
                            //   ((T - 1) * dim_action)
  std::vector<double> cxx;  // cost Hessian wrt state
                            //   (T * dim_state_derivative *
                            //    dim_state_derivative)
  std::vector<double> cuu;  // cost Hessian wrt action
                            //   ((T - 1) * dim_action * dim_action)
  std::vector<double> cxu;  // cost Hessian wrt state/action
                            //   ((T - 1) * dim_state_derivative * dim_action)

 private:
  // scratch spaces
  std::vector<double> c_scratch_;    // (T * dim_max * dim_max)
  std::vector<double> cx_scratch_;   // (T * dim_state_derivative)
  std::vector<double> cu_scratch_;   // (T * dim_action)
  std::vector<double> cxx_scratch_;  // (T * dim_state_derivative *
                                     //  dim_state_derivative)
  std::vector<double> cuu_scratch_;  // (T * dim_action * dim_action)
  std::vector<double> cxu_scratch_;  // (T * dim_state_derivative * dim_action)
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_COST_DERIVATIVES_H_
