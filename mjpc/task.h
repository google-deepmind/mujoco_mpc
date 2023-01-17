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

#ifndef MJPC_TASK_H_
#define MJPC_TASK_H_

#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "norm.h"

namespace mjpc {

// tolerance for risk-neutral cost
inline constexpr double kRiskNeutralTolerance = 1.0e-6;

// maximum cost terms
inline constexpr int kMaxCostTerms = 30;

class Task;

using ResidualFunction = void(const double* parameters, const mjModel* model,
                              const mjData* data, double* residual);
using TransitionFunction = int(int state, const mjModel* model, mjData* data,
                               Task* task);

// contains information for computing costs
class Task {
 public:
  // constructor
  Task() = default;

  // destructor
  ~Task() = default;

  // ----- methods ----- //

  // initialize task from model
  void Set(const mjModel* model, ResidualFunction* residual,
           TransitionFunction* transition);

  // get information from model
  void GetFrom(const mjModel* model);

  // compute cost terms
  void CostTerms(double* terms, const double* residual) const;

  // compute weighted cost
  double CostValue(const double* residual) const;

  // compute residuals
  void Residuals(const mjModel* m, const mjData* d, double* residuals) const;

  // apply transition function
  void Transition(const mjModel* m, mjData* d);

  int id = 0;             // task ID
  int transition_state;   // state
  int transition_status;  // status

  // cost parameters
  int num_residual;
  int num_cost;
  int num_trace;
  std::vector<int> dim_norm_residual;
  std::vector<int> num_norm_parameter;
  std::vector<NormType> norm;
  std::vector<double> weight;
  std::vector<double> num_parameter;
  double risk;

  // residual parameters
  std::vector<double> residual_parameters;

 private:
  // initial residual parameters from model
  void SetFeatureParameters(const mjModel* model);

  // residual function
  ResidualFunction* residual_;

  // transition function
  TransitionFunction* transition_;
};

extern int NullTransition(int state, const mjModel* model, mjData* data,
                          Task* task);

template <typename T = std::string>
struct TaskDefinition {
  T name;
  T xml_path;
  ResidualFunction* residual;
  TransitionFunction* transition = &NullTransition;
};

}  // namespace mjpc

#endif  // MJPC_TASK_H_
