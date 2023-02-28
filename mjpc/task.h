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
#include "mjpc/norm.h"

namespace mjpc {

// tolerance for risk-neutral cost
inline constexpr double kRiskNeutralTolerance = 1.0e-6;

// maximum cost terms
inline constexpr int kMaxCostTerms = 35;

class Task {
 public:
  // constructor
  Task() = default;
  virtual ~Task() = default;

  // ----- methods ----- //

  virtual void Residual(const mjModel* model, const mjData* data,
                        double* residual) const = 0;

  virtual void Transition(const mjModel* model, mjData* data) {}

  // get information from model
  virtual void Reset(const mjModel* model);

  virtual void ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {}

  // compute cost terms
  void CostTerms(double* terms, const double* residual,
                 bool weighted = true) const;

  // compute weighted cost
  double CostValue(const double* residual) const;

  virtual std::string Name() const = 0;
  virtual std::string XmlPath() const = 0;

  // stage
  int stage;

  // GUI toggles
  int reset = 0;
  int visualize = 0;

  // cost parameters
  int num_residual;
  int num_term;
  int num_trace;
  std::vector<int> dim_norm_residual;
  std::vector<int> num_norm_parameter;
  std::vector<NormType> norm;
  std::vector<double> weight;
  std::vector<double> num_parameter;
  double risk;

  // residual parameters
  std::vector<double> parameters;

 private:
  // initial residual parameters from model
  void SetFeatureParameters(const mjModel* model);
};

}  // namespace mjpc

#endif  // MJPC_TASK_H_
