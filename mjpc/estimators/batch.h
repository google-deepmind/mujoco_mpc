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

#ifndef MJPC_ESTIMATORS_BATCH_H_
#define MJPC_ESTIMATORS_BATCH_H_

#include <mujoco/mujoco.h>

#include <vector>

#include "mjpc/norm.h"
#include "mjpc/utilities.h"

namespace mjpc {

const int MAX_HISTORY = 32;  // maximum configuration trajectory length
const double MAX_ESTIMATOR_COST = 1.0e6; // maximum total cost

// linear system solver 
enum BatchEstimatorSolver: int {
  kCholeskyDense = 0,
  kBanded, // TODO(taylor)
};

// batch estimator
// based on: "Physically-Consistent Sensor Fusion in Contact-Rich Behaviors"
class Estimator {
 public:
  // constructor
  Estimator() {}

  // destructor
  ~Estimator() {}

  // initialize
  void Initialize(mjModel* model);

  // reset memory
  void Reset();

  // prior cost
  double CostPrior(double* gradient, double* hessian);

  // prior residual
  void ResidualPrior();

  // prior Jacobian
  void JacobianPrior();

  // prior Jacobian blocks
  void JacobianPriorBlocks();

  // sensor cost
  double CostSensor(double* gradient, double* hessian);

  // sensor residual
  void ResidualSensor();

  // sensor Jacobian
  void JacobianSensor();

  // compute sensor predictions
  void SensorPrediction();

  // force cost
  double CostForce(double* gradient, double* hessian);

  // force residual
  void ResidualForce();

  // force Jacobian
  void JacobianForce();

  // compute force predictions
  void ForcePrediction();

  // compute model derivatives (via finite difference)
  void ModelDerivatives();

  // update configuration trajectory
  void UpdateConfiguration(double* candidate, const double* configuration,
                           const double* search_direction, double step_size);

  // convert sequence of configurations to velocities, accelerations
  void ConfigurationToVelocityAcceleration();

  // compute finite-difference velocity derivatives
  void VelocityDerivatives();

  // compute finite-difference acceleration derivatives
  void AccelerationDerivatives();

  // compute total cost
  double Cost(double& cost_prior, double& cost_sensor, double& cost_force);

  // compute total cost derivatives 
  void CostDerivatives();

  // optimize trajectory estimate 
  void Optimize();

  // model
  mjModel* model_;

  // data
  mjData* data_;

  // trajectories
  int configuration_length_;                 // T
  std::vector<double> configuration_;        // nq x MAX_HISTORY
  std::vector<double> velocity_;             // nv x MAX_HISTORY
  std::vector<double> acceleration_;         // na x MAX_HISTORY
  std::vector<double> time_;                 // 1  x MAX_HISTORY

  // prior 
  std::vector<double> configuration_prior_;  // nq x MAX_HISTORY

  // sensor
  int dim_sensor_;                           // ns
  std::vector<double> sensor_measurement_;   // ns x MAX_HISTORY
  std::vector<double> sensor_prediction_;    // ns x MAX_HISTORY

  // forces
  std::vector<double> force_measurement_;    // nv x MAX_HISTORY
  std::vector<double> force_prediction_;     // nv x MAX_HISTORY

  // residual
  std::vector<double> residual_prior_;       // nv * T
  std::vector<double> residual_sensor_;      // ns * (T - 2)
  std::vector<double> residual_force_;       // nv * (T - 2)

  // Jacobian
  std::vector<double> jacobian_prior_;       // (nv * T) * (nv * T)
  std::vector<double> jacobian_sensor_;      // (ns * (T - 2)) * (nv * T)
  std::vector<double> jacobian_force_;       // (nv * (T - 2)) * (nv * T)

  // prior Jacobian block
  std::vector<double> block_prior_configuration_;  // (nv * nv) x MAX_HISTORY

  // sensor Jacobian blocks (dqds, dvds, dads)
  std::vector<double> block_sensor_configuration_; // (nv * ns) x MAX_HISTORY
  std::vector<double> block_sensor_velocity_;      // (nv * ns) x MAX_HISTORY
  std::vector<double> block_sensor_acceleration_;  // (na * ns) x MAX_HISTORY
  std::vector<double> block_sensor_scratch_;       // (nv * ns) x MAX_HISTORY

  // force Jacobian blocks (dqdf, dvdf, dadf)
  std::vector<double> block_force_configuration_;  // (nv * nv) x MAX_HISTORY
  std::vector<double> block_force_velocity_;       // (nv * nv) x MAX_HISTORY
  std::vector<double> block_force_acceleration_;   // (nv * nv) x MAX_HISTORY
  std::vector<double> block_force_scratch_;        // (nv * nv) x MAX_HISTORY

  // velocity Jacobian blocks (dv1dq0, dv1dq1)
  std::vector<double> block_velocity_previous_configuration_; // (nv * nv) x MAX_HISTORY
  std::vector<double> block_velocity_current_configuration_;  // (nv * nv) x MAX_HISTORY

  // acceleration Jacobian blocks (da1dq0, da1dq1, da1dq2)
  std::vector<double> block_acceleration_previous_configuration_; // (nv * nv) x MAX_HISTORY
  std::vector<double> block_acceleration_current_configuration_;  // (nv * nv) x MAX_HISTORY
  std::vector<double> block_acceleration_next_configuration_;     // (nv * nv) x MAX_HISTORY

  // cost 
  double cost_prior_;
  double cost_sensor_;
  double cost_force_; 
  double cost_;

  // cost gradient
  std::vector<double> cost_gradient_prior_;    // nv * MAX_HISTORY
  std::vector<double> cost_gradient_sensor_;   // nv * MAX_HISTORY
  std::vector<double> cost_gradient_force_;    // nv * MAX_HISTORY
  std::vector<double> cost_gradient_;          // nv * MAX_HISTORY

  // cost Hessian
  std::vector<double> cost_hessian_prior_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> cost_hessian_sensor_;    // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> cost_hessian_force_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> cost_hessian_;           // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)

  // cost scratch
  std::vector<double> cost_scratch_prior_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> cost_scratch_sensor_;    // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> cost_scratch_force_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)

  // weight TODO(taylor): matrices
  double weight_prior_;
  double weight_sensor_;
  double weight_force_;

  // cost norms
  NormType norm_prior_;
  NormType norm_sensor_;
  NormType norm_force_;

  // cost norm parameters
  std::vector<double> norm_parameters_prior_;  
  std::vector<double> norm_parameters_sensor_;
  std::vector<double> norm_parameters_force_;

  // norm gradient
  std::vector<double> norm_gradient_prior_;    // nv * MAX_HISTORY
  std::vector<double> norm_gradient_sensor_;   // ns * MAX_HISTORY
  std::vector<double> norm_gradient_force_;    // nv * MAX_HISTORY

  // norm Hessian
  std::vector<double> norm_hessian_prior_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> norm_hessian_sensor_;    // (ns * MAX_HISTORY) * (ns * MAX_HISTORY)
  std::vector<double> norm_hessian_force_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)

  // candidate
  std::vector<double> configuration_copy_;     // nq x MAX_HISTORY

  // search direction
  std::vector<double> search_direction_;       // nv * MAX_HISTORY

  // solver 
  BatchEstimatorSolver solver = kCholeskyDense;

  // settings 
  int max_line_search_ = 10;
  int max_smoother_iterations_ = 20;
  double gradient_tolerance_ = 1.0e-5;

  // finite-difference settings
  struct FiniteDifferenceSettings {
    double tolerance = 1.0e-5;
    bool flg_actuation = 0;
  } finite_difference_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_BATCH_H_
