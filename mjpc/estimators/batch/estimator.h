// Copyright 2023 DeepMind Technologies Limited
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

#ifndef MJPC_ESTIMATORS_BATCH_ESTIMATOR_H_
#define MJPC_ESTIMATORS_BATCH_ESTIMATOR_H_

#include <mujoco/mujoco.h>

#include <vector>

#include "mjpc/norm.h"
#include "mjpc/utilities.h"

namespace mjpc {

const int MAX_HISTORY = 32;  // maximum length configuration trajectory

// convert sequence of configurations to velocities
void ConfigurationToVelocity(double* velocity, const double* configuration,
                             int configuration_length, const mjModel* model);

// convert sequence of configurations to accelerations
void VelocityToAcceleration(double* acceleration, const double* velocity,
                            int velocity_length, const mjModel* model);

class Estimator {
 public:
  // constructor
  Estimator() {}

  // destructor
  ~Estimator() {}

  // initialize
  void Initialize(mjModel* model) {
    // model
    model_ = model;

    // data
    data_ = mj_makeData(model);

    // trajectories
    configuration_length_ = GetNumberOrDefault(10, model, "batch_length");
    configuration_.resize(model->nq * MAX_HISTORY);
    configuration_prior_.resize(model->nq * MAX_HISTORY);
    configuration_copy_.resize(model->nq * MAX_HISTORY);
    velocity_.resize(model->nv * MAX_HISTORY);
    acceleration_.resize(model->nv * MAX_HISTORY);

    // measurement
    dim_measurement_ = model->nsensordata;
    measurement_sensor_.resize(dim_measurement_ * MAX_HISTORY);
    measurement_model_.resize(dim_measurement_ * MAX_HISTORY);

    // qfrc
    qfrc_actuator_.resize(model->nv * MAX_HISTORY);
    qfrc_inverse_.resize(model->nv * MAX_HISTORY);

    // residual
    residual_prior_.resize(model->nv * MAX_HISTORY);
    residual_measurement_.resize(dim_measurement_ * MAX_HISTORY);
    residual_inverse_dynamics_.resize(model->nv * MAX_HISTORY);

    // Jacobian
    jacobian_prior_.resize((model->nv * MAX_HISTORY) *
                           (model->nv * MAX_HISTORY));
    jacobian_measurement_.resize((dim_measurement_ * MAX_HISTORY) *
                                 (model->nv * MAX_HISTORY));
    jacobian_inverse_dynamics_.resize((model->nv * MAX_HISTORY) *
                                      (model->nv * MAX_HISTORY));

    // cost gradient
    cost_gradient_prior_.resize(model->nv * MAX_HISTORY);
    cost_gradient_measurement_.resize(model->nv * MAX_HISTORY);
    cost_gradient_inverse_dynamics_.resize(model->nv * MAX_HISTORY);
    cost_gradient_total_.resize(model->nv * MAX_HISTORY);

    // cost Hessian
    cost_hessian_prior_.resize((model->nv * MAX_HISTORY) *
                               (model->nv * MAX_HISTORY));
    cost_hessian_measurement_.resize((model->nv * MAX_HISTORY) *
                                     (model->nv * MAX_HISTORY));
    cost_hessian_inverse_dynamics_.resize((model->nv * MAX_HISTORY) *
                                          (model->nv * MAX_HISTORY));
    cost_hessian_total_.resize((model->nv * MAX_HISTORY) *
                               (model->nv * MAX_HISTORY));

    // weight TODO(taylor): matrices
    weight_prior_ = GetNumberOrDefault(1.0, model, "batch_weight_prior");
    weight_measurement_ =
        GetNumberOrDefault(1.0, model, "batch_weight_measurement");
    weight_inverse_dynamics_ =
        GetNumberOrDefault(1.0, model, "batch_weight_inverse_dynamics");

    // cost norms
    norm_prior_ = (NormType)GetNumberOrDefault(0, model, "batch_norm_prior");
    norm_measurement_ =
        (NormType)GetNumberOrDefault(0, model, "batch_norm_prior");
    norm_inverse_dynamics_ =
        (NormType)GetNumberOrDefault(1.0, model, "batch_norm_prior");

    // cost norm parameters
    norm_parameters_prior_.resize(3);
    norm_parameters_measurement_.resize(3);
    norm_parameters_inverse_dynamics_.resize(3);

    // norm gradient
    norm_gradient_prior_.resize(model->nv * MAX_HISTORY);
    norm_gradient_measurement_.resize(dim_measurement_ * MAX_HISTORY);
    norm_gradient_inverse_dynamics_.resize(model->nv * MAX_HISTORY);

    // norm Hessian
    norm_hessian_prior_.resize((model->nv * MAX_HISTORY) *
                               (model->nv * MAX_HISTORY));
    norm_hessian_measurement_.resize((dim_measurement_ * MAX_HISTORY) *
                                     (dim_measurement_ * MAX_HISTORY));
    norm_hessian_inverse_dynamics_.resize((model->nv * MAX_HISTORY) *
                                          (model->nv * MAX_HISTORY));

    // update
    update_.resize(model->nv * MAX_HISTORY);
  }

  // prior cost
  double CostPrior(double* gradient, double* hessian) {
    // residual dimension
    int dim = model_->nv * configuration_length_;

    // compute cost
    double cost = Norm(gradient ? norm_gradient_prior_.data() : NULL,
                       hessian ? norm_hessian_prior_.data() : NULL,
                       residual_prior_.data(), norm_parameters_prior_.data(),
                       dim, norm_prior_);

    // compute cost gradient wrt update
    if (gradient) {
    }

    // compute cost Hessian wrt update
    if (hessian) {
    }

    return cost;
  }

  // prior residual
  void ResidualPrior() {
    for (int t = 0; t < configuration_length_; t++) {
      // terms
      double* rt = residual_prior_.data() + t * model_->nv;
      double* qt_prior = configuration_prior_.data() + t * model_->nq;
      double* qt = configuration_.data() + t * model_->nq;

      // configuration difference
      mj_differentiatePos(model_, rt, 1.0, qt_prior, qt);
    }
  }

  // prior Jacobian
  void JacobianPrior() {}

  // measurement cost
  double CostMeasurement(double* gradient, double* hessian) {
    // residual dimension
    int dim = dim_measurement_ * (configuration_length_ - 2);

    // compute cost
    double cost =
        Norm(gradient ? norm_gradient_measurement_.data() : NULL,
             hessian ? norm_hessian_measurement_.data() : NULL,
             residual_measurement_.data(), norm_parameters_measurement_.data(),
             dim, norm_measurement_);

    // cost gradient wrt update
    if (gradient) {
    }

    // cost Hessian wrt update
    if (hessian) {
    }

    return cost;
  }

  // measurement residual
  void ResidualMeasurement() {
    for (int t = 0; t < configuration_length_ - 2; t++) {
      // terms
      double* rt = residual_measurement_.data() + t * dim_measurement_;
      double* yt_sensor = measurement_sensor_.data() + t * dim_measurement_;
      double* yt_model = measurement_model_.data() + t * dim_measurement_;

      // measurement difference
      mju_sub(rt, yt_model, yt_sensor, dim_measurement_);
    }
  }

  // measurement Jacobian
  void JacobianMeasurement() {}

  // compute measurements
  void ComputeMeasurements() {
    for (int t = 0; t < configuration_length_ - 2; t++) {
      // terms
      double* qt = configuration_.data() + (t + 2) * model_->nq;
      double* vt = velocity_.data() + (t + 1) * model_->nv;
      double* at = acceleration_.data() + t * model_->nv;

      // set qt, vt, at
      mju_copy(data_->qpos, qt, model_->nq);
      mju_copy(data_->qvel, vt, model_->nv);
      mju_copy(data_->qacc, at, model_->nv);

      // sensors
      mj_inverse(model_, data_);

      // copy sensor data
      double* yt = measurement_model_.data() + t * dim_measurement_;
      mju_copy(yt, data_->sensordata, dim_measurement_);
    }
  }

  // inverse dynamics cost
  double CostInverseDynamics(double* gradient, double* hessian) {
    // residual dimension
    int dim = model_->nv * (configuration_length_ - 2);

    // compute cost
    double cost = Norm(gradient ? norm_gradient_inverse_dynamics_.data() : NULL,
                       hessian ? norm_hessian_inverse_dynamics_.data() : NULL,
                       residual_inverse_dynamics_.data(),
                       norm_parameters_inverse_dynamics_.data(), dim,
                       norm_inverse_dynamics_);

    // compute cost gradient wrt update
    if (gradient) {
    }

    // compute cost Hessian wrt update
    if (hessian) {
    }

    return cost;
  }

  // inverse dynamics residual
  void ResidualInverseDynamics() {
    for (int t = 0; t < configuration_length_ - 2; t++) {
      // terms
      double* rt = residual_inverse_dynamics_.data() + t * model_->nv;
      double* ft_actuator = qfrc_actuator_.data() + t * model_->nv;
      double* ft_inverse_ = qfrc_inverse_.data() + t * model_->nv;

      // qfrc difference
      mju_sub(rt, ft_inverse_, ft_actuator, model_->nv);
    }
  }

  // inverse dynamics Jacobian
  void JacobianInverseDynamics() {}

  // compute inverse dynamics
  void ComputeInverseDynamics() {
    for (int t = 0; t < configuration_length_ - 2; t++) {
      // terms
      double* qt = configuration_.data() + (t + 2) * model_->nq;
      double* vt = velocity_.data() + (t + 1) * model_->nv;
      double* at = acceleration_.data() + t * model_->nv;

      // set qt, vt, at
      mju_copy(data_->qpos, qt, model_->nq);
      mju_copy(data_->qvel, vt, model_->nv);
      mju_copy(data_->qacc, at, model_->nv);

      // inverse dynamics
      mj_inverse(model_, data_);

      // copy qfrc
      double* ft = qfrc_inverse_.data() + t * model_->nv;
      mju_copy(ft, data_->qfrc_inverse, model_->nv);
    }
  }

  void UpdateConfiguration(double* configuration, const double* update) {
    for (int t = 0; t < configuration_length_; t++) {
      // configuration
      double* q = configuration + t * model_->nq;

      // update
      const double* dq = update + t * model_->nv;

      // integrate
      mj_integratePos(model_, q, dq, 1.0);
    }
  }

  void UpdateTrajectory(double* configuration, const double* update) {
    // update configuration trajectory using
    UpdateConfiguration(configuration, update);

    // finite-difference velocities
    // ConfigurationToVelocity(velocity_, configuration, configuration_length_,
    //                         model_);

    // // finite-difference accelerations
    // VelocityToAcceleration(accleration_, velocity_, configuration_length_ -
    // 1,
    //                        model_);

    // compute model measurements
    ComputeMeasurements();

    // compute model qfrc
    ComputeInverseDynamics();
  }

  // model
  mjModel* model_;

  // data
  mjData* data_;

  // trajectories
  int configuration_length_;
  std::vector<double> configuration_;
  std::vector<double> configuration_prior_;
  std::vector<double> configuration_copy_;
  std::vector<double> velocity_;
  std::vector<double> acceleration_;

  // measurements
  int dim_measurement_;
  std::vector<double> measurement_sensor_;
  std::vector<double> measurement_model_;

  // qfrc
  std::vector<double> qfrc_actuator_;
  std::vector<double> qfrc_inverse_;

  // residual
  std::vector<double> residual_prior_;
  std::vector<double> residual_measurement_;
  std::vector<double> residual_inverse_dynamics_;

  // Jacobian
  std::vector<double> jacobian_prior_;
  std::vector<double> jacobian_measurement_;
  std::vector<double> jacobian_inverse_dynamics_;

  // cost gradient
  std::vector<double> cost_gradient_prior_;
  std::vector<double> cost_gradient_measurement_;
  std::vector<double> cost_gradient_inverse_dynamics_;
  std::vector<double> cost_gradient_total_;

  // cost Hessian
  std::vector<double> cost_hessian_prior_;
  std::vector<double> cost_hessian_measurement_;
  std::vector<double> cost_hessian_inverse_dynamics_;
  std::vector<double> cost_hessian_total_;

  // weight TODO(taylor): matrices
  double weight_prior_;
  double weight_measurement_;
  double weight_inverse_dynamics_;

  // cost norms
  NormType norm_prior_;
  NormType norm_measurement_;
  NormType norm_inverse_dynamics_;

  // cost norm parameters
  std::vector<double> norm_parameters_prior_;
  std::vector<double> norm_parameters_measurement_;
  std::vector<double> norm_parameters_inverse_dynamics_;

  // norm gradient
  std::vector<double> norm_gradient_prior_;
  std::vector<double> norm_gradient_measurement_;
  std::vector<double> norm_gradient_inverse_dynamics_;

  // norm Hessian
  std::vector<double> norm_hessian_prior_;
  std::vector<double> norm_hessian_measurement_;
  std::vector<double> norm_hessian_inverse_dynamics_;

  // update
  std::vector<double> update_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_BATCH_ESTIMATOR_H_
