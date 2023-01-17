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

#include "task.h"

#include <cstring>

#include <mujoco/mujoco.h>
#include "utilities.h"

namespace mjpc {

// initialize task from model
void Task::Set(const mjModel* model, ResidualFunction* residual,
               TransitionFunction* transition) {
  // get information from model
  this->GetFrom(model);

  // set residual function
  this->residual_ = residual;

  // set transition function
  this->transition_ = transition;
}

void Task::GetFrom(const mjModel* model) {
  // ----- defaults ----- //

  // transition
  transition_status = GetNumberOrDefault(0, model, "task_transition");
  transition_state = 0;

  // risk value
  risk = GetNumberOrDefault(0.0, model, "task_risk");

  // set residual parameters
  this->SetFeatureParameters(model);

  // ----- set costs ----- //
  num_cost = 0;
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
      dim_norm_residual[num_cost] = (int)model->sensor_dim[i];

      // user data: [norm, weight, weight_lower, weight_upper, parameters...]
      double* s = model->sensor_user + i * model->nuser_sensor;

      // check number of parameters
      for (int j = 0; j < NormParameterDimension(s[0]); j++) {
        if (s[4 + j] > 0.0) continue;
        mju_error("Cost construction from XML: Missing parameter value\n");
      }
      norm[num_cost] = (NormType)s[0];
      weight[num_cost] = s[1];
      num_norm_parameter[num_cost] = NormParameterDimension(s[0]);
      mju_copy(DataAt(num_parameter, parameter_shift), s + 4,
               num_norm_parameter[num_cost]);
      parameter_shift += num_norm_parameter[num_cost];
      num_cost += 1;

      // check for max norms
      if (num_cost > kMaxCostTerms) {
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
void Task::CostTerms(double* terms, const double* residual) const {
  int f_shift = 0;
  int p_shift = 0;
  for (int k = 0; k < num_cost; k++) {
    // running cost
    terms[k] = weight[k] * Norm(nullptr, nullptr, residual + f_shift,
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
  for (int i = 0; i < num_cost; i++) {
    cost += terms[i];
  }

  // exponential risk transformation
  if (mju_abs(risk) < kRiskNeutralTolerance) {
    return cost;
  } else {
    return (mju_exp(risk * cost) - 1.0) / risk;
  }
}

void Task::Residuals(const mjModel* m, const mjData* d,
                     double* residuals) const {
  residual_(residual_parameters.data(), m, d, residuals);
}

void Task::Transition(const mjModel* m, mjData* d) {
  transition_state = transition_(transition_state, m, d, this);
}

// initial residual parameters from model
void Task::SetFeatureParameters(const mjModel* model) {
  // set counter
  int num_residual_parameters = 0;

  // search custom numeric in model for "residual"
  for (int i = 0; i < model->nnumeric; i++) {
    if (std::strncmp(model->names + model->name_numericadr[i], "residual_",
                     8) == 0) {
      num_residual_parameters += 1;
    }
  }

  // allocate memory
  residual_parameters.resize(num_residual_parameters);

  // set values
  int shift = 0;
  for (int i = 0; i < model->nnumeric; i++) {
    if (std::strncmp(model->names + model->name_numericadr[i], "residual_",
                     9) == 0) {
      int dim = 1;  // model->numeric_size[i];
      double* params = model->numeric_data + model->numeric_adr[i];
      mju_copy(DataAt(residual_parameters, shift), params, dim);
      shift += dim;
    }
  }
}

int NullTransition(int state, const mjModel* model, mjData* data, Task* task) {
  return state;
}

}  // namespace mjpc
