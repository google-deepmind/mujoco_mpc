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

#include "mjpc/task.h"

#include <cstring>

#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

// called at: construction, load, and GUI reset
void Task::Reset(const mjModel* model) {
  // ----- defaults ----- //

  // transition
  stage = 0;

  // risk value
  risk = GetNumberOrDefault(0.0, model, "task_risk");

  // set residual parameters
  this->SetFeatureParameters(model);

  // ----- set costs ----- //
  num_term = 0;
  num_residual = 0;
  num_trace = 0;

  // allocate memory
  dim_norm_residual.resize(kMaxCostTerms);
  num_norm_parameter.resize(kMaxCostTerms);
  norm.resize(kMaxCostTerms);
  weight.resize(kMaxCostTerms);
  num_parameter.resize(2 * kMaxCostTerms);

  // check user sensor is first
  if (!(model->sensor_type[0] == mjSENS_USER)) {
    mju_error(
        "Cost construction from XML: User sensors specifying residuals must be "
        "specified first and sequentially\n");
  }

  // get number of traces
  for (int i = 0; i < model->nsensor; i++) {
    if (std::strncmp(model->names + model->name_sensoradr[i], "trace",
                     5) == 0) {
      num_trace += 1;
    }
  }
  if (num_trace > kMaxTraces) {
    mju_error("Number of traces should be less than 100\n");
  }

  // loop over sensors
  int parameter_shift = 0;
  for (int i = 0; i < model->nsensor; i++) {
    if (model->sensor_type[i] == mjSENS_USER) {
      // residual dimension
      num_residual += model->sensor_dim[i];
      dim_norm_residual[num_term] = (int)model->sensor_dim[i];

      // user data: [norm, weight, weight_lower, weight_upper, parameters...]
      double* s = model->sensor_user + i * model->nuser_sensor;

      // check number of parameters
      for (int j = 0; j < NormParameterDimension(s[0]); j++) {
        if (s[4 + j] > 0.0) continue;
        mju_error("Cost construction from XML: Missing parameter value\n");
      }
      norm[num_term] = (NormType)s[0];

      // check Null norm
      if (norm[num_term] == -1 && dim_norm_residual[num_term] != 1) {
        mju_error("Cost construction from XML: Missing parameter value\n");
      }

      weight[num_term] = s[1];
      num_norm_parameter[num_term] = NormParameterDimension(s[0]);
      mju_copy(DataAt(num_parameter, parameter_shift), s + 4,
               num_norm_parameter[num_term]);
      parameter_shift += num_norm_parameter[num_term];
      num_term += 1;

      // check for max norms
      if (num_term > kMaxCostTerms) {
        mju_error(
            "Number of cost terms exceeds maximum. Either: 1) reduce number of "
            "terms 2) increase kMaxCostTerms");
      }
    }
  }

  // set residual parameters
  this->SetFeatureParameters(model);
}

// compute weighted cost terms
void Task::CostTerms(double* terms, const double* residual,
                     bool weighted) const {
  int f_shift = 0;
  int p_shift = 0;
  for (int k = 0; k < num_term; k++) {
    // running cost
    terms[k] =
        (weighted ? weight[k] : 1) * Norm(nullptr, nullptr, residual + f_shift,
                                          DataAt(num_parameter, p_shift),
                                          dim_norm_residual[k], norm[k]);

    // shift residual
    f_shift += dim_norm_residual[k];

    // shift parameters
    p_shift += num_norm_parameter[k];
  }
}

// compute weighted cost from terms
double Task::CostValue(const double* residual) const {
  // cost terms
  double terms[kMaxCostTerms];

  // evaluate
  this->CostTerms(terms, residual);

  // summation of cost terms
  double cost = 0.0;
  for (int i = 0; i < num_term; i++) {
    cost += terms[i];
  }

  // exponential risk transformation
  if (mju_abs(risk) < kRiskNeutralTolerance) {
    return cost;
  } else {
    return (mju_exp(risk * cost) - 1.0) / risk;
  }
}

// initial residual parameters from model
void Task::SetFeatureParameters(const mjModel* model) {
  // set counter
  int num_parameters = 0;

  // search custom numeric in model for "residual"
  for (int i = 0; i < model->nnumeric; i++) {
    if (absl::StartsWith(model->names + model->name_numericadr[i],
                         "residual_")) {
      num_parameters += 1;
    }
  }

  // allocate memory
  parameters.resize(num_parameters);

  // set values
  int shift = 0;
  for (int i = 0; i < model->nnumeric; i++) {
    if (absl::StartsWith(model->names + model->name_numericadr[i],
                         "residual_select_")) {
      parameters[shift++] = DefaultResidualSelection(model, i);
    } else if (absl::StartsWith(model->names + model->name_numericadr[i],
                                "residual_")) {
      parameters[shift++] = model->numeric_data[model->numeric_adr[i]];
    }
  }
}
}  // namespace mjpc
