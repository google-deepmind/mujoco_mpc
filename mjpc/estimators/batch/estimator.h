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

#include "mjpc/utilities.h"

namespace mjpc {

const int MAX_HISTORY = 128;

class Estimator {
 public:
  // constructor
  Estimator() {}

  // destructor
  ~Estimator() {}

  // // initialize
  // void Initialize(mjModel* model) {
  //   // model
  //   model_ = model;

  //   // data
  //   data_ = mj_makeData(model);

  //   // trajectories
  //   configuration_length_ = GetNumberOrDefault(10, model, "batch_length");
  //   configuration_.resize(model->nq * MAX_HISTORY);
  //   configuration_prior_.resize(model->nq * MAX_HISTORY);
  //   configuration_copy_.resize(model->nq * MAX_HISTORY);
  //   velocity_.resize(model->nv * MAX_HISTORY);
  //   acceleration_.resize(model->nv * MAX_HISTORY);

  //   // measurement
  //   dim_measurement_ = model->nsensordata;
  //   measurement_sensor_.resize(dim_measurement * MAX_HISTORY);
  //   measurement_model_.resize(dim_measurement * MAX_HISTORY);

  //   // actuation
  //   qfrc_actuator_sensor_.resize(model->nv * MAX_HISTORY);
  //   qfrc_actuator_model_.resize(model->nv * MAX_HISTORY);

  //   // cost
  //   residual_prior_.resize(model->nv * MAX_HISTORY);
  //   residual_measurement_.resize(dim_measurement_ * MAX_HISTORY);
  //   residual_inverse_dynamics_.resize(model->nv * MAX_HISTORY);

  //   // update 
  //   update_.resize(model->nv * MAX_HISTORY);
  // }

  // // prior cost
  // double CostPrior();

  // // prior residual
  // void ResidualPrior() {
  //   for (int t = 0; t < configuration_length_; t++) {
  //     // terms
  //     double* rt = residual_prior_.data() + t * model_->nv;
  //     double* qt_prior = configuration_prior_.data() + t * model->nq;
  //     double* qt = configuration_.data() + t * model->nq;

  //     // configuration difference
  //     mj_differentiatePos(model_, rt, 1.0, qt_prior, qt);
  //   }

  //   // measurement cost
  //   double CostMeasurement();

  //   // measurement residual
  //   void ResidualMeasurement() {
  //     for (int t = 0; t < configuration_length_ - 2; t++) {
  //       // terms
  //       double* rt = residual_measurement_.data() + t * dim_measurement_;
  //       double* yt_sensor = measurement_sensor_.data() + t * dim_measurement_;
  //       double* yt_model = measurement_model_.data() + t * dim_measurement_;

  //       // measurement difference
  //       mju_sub(rt, yt_model, yt_sensor, dim_measurement_);
  //     }
  //   }

  //   // compute measurements
  //   void ComputeMeasurements() {
  //     for (int t = 0; t < configuration_ - 2; t++) {
  //       // terms
  //       double qt = configuration_.data() + (t + 2) * model_->nq;
  //       double vt = velocity_.data() + (t + 1) * model_->nv;
  //       double at = acceleration_.data() + t * model_->nv;

  //       // set qt, vt, at
  //       mju_copy(data_->qpos, qt, model_->nq);
  //       mju_copy(data_->qvel, vt, model_->nv);
  //       mju_copy(data_->qacc, at, model_->nv);
  //       // TODO(taylor) set ctrl / qfrc_actuator?

  //       // sensors
  //       mj_forward(model_, data_); // TODO(taylor) is this correct ? 

  //       // copy sensor data
  //       double* yt = measurements_model_.data() + t * dim_measurement_;
  //       mju_copy(yt, data_->sensordata, dim_measurement_);
  //     }
  //   }

  //   // inverse dynamics cost
  //   double CostInverseDynamics();

  //   // inverse dynamics residual
  //   void ResidualInverseDynamics() {
  //     for (int t = 0; t < configuration_length - 2; t++) {
  //       // terms
  //       double* rt = residual_inverse_dynamics_.data() + t * model_->nv;
  //       double* ft_actuator = qfrc_actuator_.data() + t * model->nv;
  //       double* ft_model = qfrc_model_.data() + t * model->nv;

  //       // actuation difference
  //       mju_sub(rt, ft_model, ft_actuator, model_->nv);
  //     }
  //   }

  //   // compute inverse dynamics 
  //   void ComputeInverseDynamics() {
  //     for (int t = 0; t < configuration_length - 2; t++) {
  //       // terms
  //       double qt = configuration_.data() + (t + 2) * model_->nq;
  //       double vt = velocity_.data() + (t + 1) * model_->nv;
  //       double at = acceleration_.data() + t * model_->nv;

  //       // set qt, vt, at
  //       mju_copy(data_->qpos, qt, model_->nq);
  //       mju_copy(data_->qvel, vt, model_->nv);
  //       mju_copy(data_->qacc, at, model_->nv);
  //       // TODO(taylor): set ctrl ?

  //       // inverse dynamics 
  //       mj_inverse(model_, data_);

  //       // copy qfrc
  //       double* ft = qfrc_model_.data() + t * model_->nv;
  //       mju_copy(ft, data_->qfrc, model_->nv);
  //     }
  //   }

  //   void UpdateConfiguration(double* configuration) {
  //     for (int t = 0; t < configuration_length_; t++) {
  //       // configuration
  //       double* q = configuration.data() + t * model_->nq;

  //       // update
  //       double* dq = update_.data() + t * model_->nv;

  //       // integrate
  //       mj_integratePos(model_, q, dq, 1.0);
  //     }
  //   }

  //   void UpdateTrajectory(double* configuration) {
  //     // update configuration trajectory using 
  //     UpdateConfiguration(configuration);

  //     // finite-difference velocities
  //     ConfigurationToVelocity(velocity_, configuration, configuration_length_,
  //                             model_);

  //     // finite-difference accelerations
  //     VelocityToAcceleration(accleration_, velocity_, configuration_length_ - 1,
  //                            model_);

  //     // compute model measurements 
  //     ComputeMeasurements();

  //     // compute model qfrc
  //     ComputeInverseDynamics();
  //   }

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

    // actuation
    std::vector<double> qfrc_actuator_;
    std::vector<double> qfrc_model_;

    // cost
    std::vector<double> residual_prior_;
    std::vector<double> residual_measurement_;
    std::vector<double> residual_inverse_dynamics_;

    // update 
    std::vector<double> update_;
  };

  // convert sequence of configurations to velocities
  void ConfigurationToVelocity(double* velocity, const double* configuration,
                               int configuration_length, const mjModel* model);

  // convert sequence of configurations to accelerations
  void VelocityToAcceleration(double* acceleration, const double* velocity,
                              int velocity_length, const mjModel* model);

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_BATCH_ESTIMATOR_H_
