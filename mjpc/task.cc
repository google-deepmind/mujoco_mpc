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
#include <mutex>

#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

using ForwardingResidualFn = internal::ForwardingResidualFn;

namespace {
void MissingParameterError(const mjModel* m, int sensorid) {
  mju_error(
      "Cost construction from XML: Missing parameter value."
      " sensor ID = %d (%s)",
      sensorid, m->names + m->name_sensoradr[sensorid]);
}
}  // namespace


Task::Task() : default_residual_(this) {}

// called at: construction, load, and GUI reset
void Task::Reset(const mjModel* model) {
  // ----- defaults ----- //

  // mode
  mode = 0;

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
  norm_parameter.resize(2 * kMaxCostTerms);

  // check user sensor is first
  if (!(model->sensor_type[0] == mjSENS_USER)) {
    mju_error(
        "Cost construction from XML: User sensors specifying residuals must be "
        "specified first and sequentially\n");
  }

  for (int i = 1; true; i++) {
    if (i == model->nsensor || model->sensor_type[i] != mjSENS_USER) {
      num_term = i;
      break;
    }
  }
  if (num_term > kMaxCostTerms) {
    mju_error(
        "Number of cost terms exceeds maximum. Either: 1) reduce number of "
        "terms 2) increase kMaxCostTerms");
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
  for (int i = 0; i < num_term; i++) {
    // residual dimension
    num_residual += model->sensor_dim[i];
    dim_norm_residual[i] = (int)model->sensor_dim[i];

    // user data: [norm, weight, weight_lower, weight_upper, parameters...]
    double* s = model->sensor_user + i * model->nuser_sensor;

    // check number of parameters
    int norm_parameter_dimension = NormParameterDimension(s[0]);
    if (4 + norm_parameter_dimension > model->nuser_sensor) {
      MissingParameterError(model, i);
      return;
    }
    for (int j = 0; j < norm_parameter_dimension; j++) {
      if (s[4 + j] <= 0.0) {
        MissingParameterError(model, i);
        return;
      }
    }
    norm[i] = (NormType)s[0];

    // check Null norm
    if (norm[i] == -1 && dim_norm_residual[i] != 1) {
      MissingParameterError(model, i);
      return;
    }

    weight[i] = s[1];
    num_norm_parameter[i] = norm_parameter_dimension;
    mju_copy(DataAt(norm_parameter, parameter_shift), s + 4,
             num_norm_parameter[i]);
    parameter_shift += num_norm_parameter[i];
  }

  // set residual parameters
  this->SetFeatureParameters(model);
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

BaseResidualFn::BaseResidualFn(const Task* task) : task_(task) {
  Update();
}

// compute weighted cost terms
void BaseResidualFn::CostTerms(double* terms, const double* residual,
                               bool weighted) const {
  int f_shift = 0;
  int p_shift = 0;
  for (int k = 0; k < num_term_; k++) {
    // running cost
    terms[k] =
        (weighted ? weight_[k] : 1) * Norm(nullptr, nullptr, residual + f_shift,
                                           DataAt(norm_parameter_, p_shift),
                                           dim_norm_residual_[k], norm_[k]);

    // shift residual
    f_shift += dim_norm_residual_[k];

    // shift parameters
    p_shift += num_norm_parameter_[k];
  }
}

// compute weighted cost from terms
double BaseResidualFn::CostValue(const double* residual) const {
  // cost terms
  double terms[kMaxCostTerms];

  // evaluate
  this->CostTerms(terms, residual, /*weighted=*/true);

  // summation of cost terms
  double cost = 0.0;
  for (int i = 0; i < num_term_; i++) {
    cost += terms[i];
  }

  // exponential risk transformation
  if (mju_abs(risk_) < kRiskNeutralTolerance) {
    return cost;
  } else {
    return (mju_exp(risk_ * cost) - 1.0) / risk_;
  }
}

void BaseResidualFn::Update() {
  num_residual_ = task_->num_residual;
  num_term_ = task_->num_term;
  num_trace_ = task_->num_trace;
  dim_norm_residual_ = task_->dim_norm_residual;
  num_norm_parameter_ = task_->num_norm_parameter;
  norm_ = task_->norm;
  weight_ = task_->weight;
  norm_parameter_ = task_->norm_parameter;
  risk_ = task_->risk;
  parameters_ = task_->parameters;
}

// default implementation calls down to Task::Residual, for backwards compat.
// this is not thread safe, but it's what most existing tasks do.
void ForwardingResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  task_->Residual(model, data, residual);
}

// compute weighted cost terms
void ForwardingResidualFn::CostTerms(double* terms, const double* residual,
                                     bool weighted) const {
  int f_shift = 0;
  int p_shift = 0;
  for (int k = 0; k < task_->num_term; k++) {
    // running cost
    terms[k] = (weighted ? task_->weight[k] : 1) *
               Norm(nullptr, nullptr, residual + f_shift,
                    DataAt(task_->norm_parameter, p_shift),
                    task_->dim_norm_residual[k], task_->norm[k]);

    // shift residual
    f_shift += task_->dim_norm_residual[k];

    // shift parameters
    p_shift += task_->num_norm_parameter[k];
  }
}

// compute weighted cost from terms
double ForwardingResidualFn::CostValue(const double* residual) const {
  // cost terms
  double terms[kMaxCostTerms];

  // evaluate
  this->CostTerms(terms, residual, /*weighted=*/true);

  // summation of cost terms
  double cost = 0.0;
  for (int i = 0; i < task_->num_term; i++) {
    cost += terms[i];
  }

  // exponential risk transformation
  if (mju_abs(task_->risk) < kRiskNeutralTolerance) {
    return cost;
  } else {
    return (mju_exp(task_->risk * cost) - 1.0) / task_->risk;
  }
}

std::unique_ptr<ResidualFn> ThreadSafeTask::Residual() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return ResidualLocked();
}

void ThreadSafeTask::Residual(const mjModel* model, const mjData* data,
                              double* residual) const {
  std::lock_guard<std::mutex> lock(mutex_);
  InternalResidual()->Residual(model, data, residual);
}

void ThreadSafeTask::UpdateResidual() {
  std::lock_guard<std::mutex> lock(mutex_);
  InternalResidual()->Update();
}

void ThreadSafeTask::Transition(const mjModel* model, mjData* data) {
  std::lock_guard<std::mutex> lock(mutex_);
  TransitionLocked(model, data, &mutex_);
  InternalResidual()->Update();
}

void ThreadSafeTask::Reset(const mjModel* model) {
  std::lock_guard<std::mutex> lock(mutex_);
  Task::Reset(model);
  ResetLocked(model);
  InternalResidual()->Update();
}

void ThreadSafeTask::CostTerms(double* terms, const double* residual,
                               bool weighted) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return InternalResidual()->CostTerms(terms, residual, weighted);
}

double ThreadSafeTask::CostValue(const double* residual) const {
  std::lock_guard<std::mutex> lock(mutex_);
  return InternalResidual()->CostValue(residual);
}

}  // namespace mjpc
