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

#ifndef MJPC_ESTIMATORS_ESTIMATOR_H_
#define MJPC_ESTIMATORS_ESTIMATOR_H_

#include <mutex>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/estimators/buffer.h"
#include "mjpc/estimators/trajectory.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

const int MIN_HISTORY = 3;    // minimum configuration trajectory length
const int MAX_HISTORY = 256;  // maximum configuration trajectory length

const int NUM_FORCE_TERMS = 3;
const int MAX_NORM_PARAMETERS = 3;

// search type for update
enum SearchType : int {
  kLineSearch = 0,
  kCurveSearch,
  kNumSearch,
};

// maximum / minimum regularization
const double MAX_REGULARIZATION = 1.0e12;
const double MIN_REGULARIZATION = 1.0e-12;

// batch estimator
// based on: "Physically-Consistent Sensor Fusion in Contact-Rich Behaviors"
class Estimator {
 public:
  // constructor
  Estimator() = default;
  Estimator(mjModel* model, int length) {
    Initialize(model);
    SetConfigurationLength(length);
  }

  // destructor
  ~Estimator() {}

  // initialize
  void Initialize(mjModel* model);

  // set configuration length
  void SetConfigurationLength(int length);

  // shift trajectory heads
  void Shift(int shift);

  // reset memory
  void Reset();

  // evaluate configurations
  void ConfigurationEvaluation(ThreadPool& pool);

  // compute total cost
  double Cost(double* gradient, double* hessian, ThreadPool& pool);

  // optimize trajectory estimate
  void Optimize(ThreadPool& pool);

  // initialize trajectories
  void InitializeTrajectories(const EstimatorTrajectory<double>& measurement,
                              const EstimatorTrajectory<int>& measurement_mask,
                              const EstimatorTrajectory<double>& ctrl,
                              const EstimatorTrajectory<double>& time);

  // update trajectories
  int UpdateTrajectories_(int num_new,
                          const EstimatorTrajectory<double>& measurement,
                          const EstimatorTrajectory<int>& measurement_mask,
                          const EstimatorTrajectory<double>& ctrl,
                          const EstimatorTrajectory<double>& time);
  int UpdateTrajectories(const EstimatorTrajectory<double>& measurement,
                         const EstimatorTrajectory<int>& measurement_mask,
                         const EstimatorTrajectory<double>& ctrl,
                         const EstimatorTrajectory<double>& time);

  // update
  int Update(const Buffer& buffer, ThreadPool& pool);

  // get configuration length 
  int ConfigurationLength() const { return configuration_length_; }
  
  // get prediction length
  int PredictionLength() const { return prediction_length_; }

  // get number of sensors 
  int NumberSensors() const { return num_sensor_; }

  // get dimension of sensors 
  int SensorDimension() const { return dim_sensor_; }

  // trajectories
  EstimatorTrajectory<double> configuration;           // nq x T
  EstimatorTrajectory<double> configuration_previous;  // nq x T
  EstimatorTrajectory<double> velocity;                // nv x T
  EstimatorTrajectory<double> acceleration;            // nv x T
  EstimatorTrajectory<double> time;                    //  1 x T
  EstimatorTrajectory<double> ctrl;                    // nu x T
  EstimatorTrajectory<double> sensor_measurement;      // ns x T
  EstimatorTrajectory<double> sensor_prediction;       // ns x T
  EstimatorTrajectory<int> sensor_mask;                // num_sensor x T
  EstimatorTrajectory<double> force_measurement;       // nv x T
  EstimatorTrajectory<double> force_prediction;        // nv x T
  
  // model
  mjModel* model;

  // cost
  double cost_prior;
  double cost_sensor;
  double cost_force;
  double cost;
  double cost_initial;

  // cost gradient
  std::vector<double> cost_gradient;          // nv * MAX_HISTORY

  // cost Hessian
  std::vector<double> cost_hessian;           // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)

  // prior weights
  std::vector<double> weight_prior;           // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> weight_prior_band;      // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)

  // scale
  double scale_prior;
  std::vector<double> scale_sensor;           // num_sensor
  std::vector<double> scale_force;            // NUM_FORCE_TERMS

  // norms
  std::vector<NormType> norm_sensor;          // num_sensor
  NormType norm_force[NUM_FORCE_TERMS];       // NUM_FORCE_TERMS

  // norm parameters
  std::vector<double> norm_parameters_sensor; // num_sensor x MAX_NORM_PARAMETERS
  std::vector<double> norm_parameters_force;  // NUM_FORCE_TERMS x MAX_NORM_PARAMETERS

  // initial state
  std::vector<double> qpos0;
  std::vector<double> qvel0;

  // status
  int iterations_smoother_;                 // total smoother iterations after Optimize
  int iterations_line_search_;              // total line search iterations
  double gradient_norm_;                    // norm of cost gradient
  double regularization_;
  double step_size_;

  // settings
  bool prior_flag = true;
  bool sensor_flag = true;
  bool force_flag = true;
  int max_line_search = 1000;                // maximum number of line search iterations
  int max_smoother_iterations = 100;        // maximum number of smoothing iterations
  double gradient_tolerance = 1.0e-10;     // small gradient tolerance
  bool verbose_optimize = false;           // flag for printing optimize status
  bool verbose_cost = false;               // flag for printing cost
  bool verbose_prior = false;              // flag for printing prior weight update status
  bool band_prior = true;                  // approximate covariance for prior
  double step_scaling = 0.5;               // step size scaling
  double regularization_initial = 1.0e-12;  // initial regularization
  double regularization_scaling = 2.0;    // regularization scaling
  bool band_copy = true;                   // copy band matrices by block
  bool reuse_data = false;                 // flag for resuing data previously computed
  bool skip_update_prior_weight = false;    // flag for skipping update prior weight
  bool update_prior_weight = true;         // flag for updating prior weights
  bool time_scaling = false;               // scale sensor and force costs by time step
  SearchType search_type;                  // search type (line search, curve search)

  // finite-difference settings
  struct FiniteDifferenceSettings {
    double tolerance = 1.0e-7;
    bool flg_actuation = 1;
  } finite_difference;

 private:
  // convert sequence of configurations to velocities, accelerations
  void ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions via inverse dynamics
  void InverseDynamicsPrediction(ThreadPool& pool);

  // compute finite-difference velocity, acceleration derivatives
  void VelocityAccelerationDerivatives();

  // compute inverse dynamics derivatives (via finite difference)
  void InverseDynamicsDerivatives(ThreadPool& pool);

  // evaluate configurations derivatives
  void ConfigurationDerivative(ThreadPool& pool);

  // prior cost
  double CostPrior(double* gradient, double* hessian);

  // sensor cost
  double CostSensor(double* gradient, double* hessian);

  // force cost
  double CostForce(double* gradient, double* hessian);

  // compute total gradient
  void TotalGradient();

  // compute total Hessian
  void TotalHessian();

  // prior residual
  void ResidualPrior();

  // set block in prior Jacobian
  void SetBlockPrior(int index);

  // prior Jacobian block
  void BlockPrior(int index);

  // prior Jacobian
  void JacobianPrior(ThreadPool& pool);

  // sensor residual
  void ResidualSensor();

  // set block in sensor Jacobian
  void SetBlockSensor(int index);

  // sensor Jacobian blocks (dsdq0, dsdq1, dsdq2)
  void BlockSensor(int index);

  // sensor Jacobian
  void JacobianSensor(ThreadPool& pool);
  
  // force residual
  void ResidualForce();

  // set block in force Jacobian
  void SetBlockForce(int index);

  // force Jacobian blocks (dfdq0, dfdq1, dfdq2)
  void BlockForce(int index);

  // force Jacobian
  void JacobianForce(ThreadPool& pool);

  // regularize Hessian
  void Regularize();

  // search direction
  void SearchDirection();

   // update configuration trajectory
  void UpdateConfiguration(EstimatorTrajectory<double>& candidate,
                           const EstimatorTrajectory<double>& configuration,
                           const double* search_direction, double step_size);

  // prior weight update
  void PriorWeightUpdate(ThreadPool& pool);

  // reset timers
  void ResetTimers();

  // print optimize status
  void PrintOptimize();

  // print cost
  void PrintCost();

  // print update prior weight status
  void PrintPriorWeightUpdate();

  // data
  std::vector<UniqueMjData> data_;

  // lengths
  int configuration_length_;                   // T
  int prediction_length_;                      // T - 2

  // dimensions
  int dim_sensor_;                                   // ns
  int num_sensor_;                                   // num_sensor 
  int num_free_;                                     // number of free joints
  std::vector<bool> free_dof_;                       // flag indicating free joint dof

  // configuration copy
  EstimatorTrajectory<double> configuration_copy_;  // nq x MAX_HISTORY

  // residual
  std::vector<double> residual_prior_;       // nv x T
  std::vector<double> residual_sensor_;      // ns x (T - 2)
  std::vector<double> residual_force_;       // nv x (T - 2)

  // Jacobian
  std::vector<double> jacobian_prior_;       // (nv * T) * (nv * T)
  std::vector<double> jacobian_sensor_;      // (ns * (T - 2)) * (nv * T)
  std::vector<double> jacobian_force_;       // (nv * (T - 2)) * (nv * T)

  // prior Jacobian block
  EstimatorTrajectory<double> block_prior_current_configuration_;  // (nv * nv) x T

  // sensor Jacobian blocks (dqds, dvds, dads), (dsdq0, dsdq1, dsdq2)
  EstimatorTrajectory<double> block_sensor_configuration_;           // (nv * ns) x T
  EstimatorTrajectory<double> block_sensor_velocity_;                // (nv * ns) x T
  EstimatorTrajectory<double> block_sensor_acceleration_;            // (nv * ns) x T

  EstimatorTrajectory<double> block_sensor_previous_configuration_;  // (ns * nv) x T
  EstimatorTrajectory<double> block_sensor_current_configuration_;   // (ns * nv) x T
  EstimatorTrajectory<double> block_sensor_next_configuration_;      // (ns * nv) x T
  EstimatorTrajectory<double> block_sensor_configurations_;          // (ns * 3 * nv) x T

  EstimatorTrajectory<double> block_sensor_scratch_;                 // max(nv, ns) x T

  // force Jacobian blocks (dqdf, dvdf, dadf), (dfdq0, dfdq1, dfdq2)
  EstimatorTrajectory<double> block_force_configuration_;            // (nv * nv) x T
  EstimatorTrajectory<double> block_force_velocity_;                 // (nv * nv) x T
  EstimatorTrajectory<double> block_force_acceleration_;             // (nv * nv) x T

  EstimatorTrajectory<double> block_force_previous_configuration_;   // (nv * nv) x T
  EstimatorTrajectory<double> block_force_current_configuration_;    // (nv * nv) x T
  EstimatorTrajectory<double> block_force_next_configuration_;       // (nv * nv) x T
  EstimatorTrajectory<double> block_force_configurations_;           // (nv * 3 * nv) x T

  EstimatorTrajectory<double> block_force_scratch_;                  // (nv * nv) x T

  // velocity Jacobian blocks (dv1dq0, dv1dq1)
  EstimatorTrajectory<double> block_velocity_previous_configuration_; // (nv * nv) x T
  EstimatorTrajectory<double> block_velocity_current_configuration_;  // (nv * nv) x T

  // acceleration Jacobian blocks (da1dq0, da1dq1, da1dq2)
  EstimatorTrajectory<double> block_acceleration_previous_configuration_; // (nv * nv) x T
  EstimatorTrajectory<double> block_acceleration_current_configuration_;  // (nv * nv) x T
  EstimatorTrajectory<double> block_acceleration_next_configuration_;     // (nv * nv) x T

  // norm gradient
  std::vector<double> norm_gradient_sensor_;   // ns * MAX_HISTORY
  std::vector<double> norm_gradient_force_;    // nv * MAX_HISTORY

  // norm Hessian
  std::vector<double> norm_hessian_sensor_;    // (ns * ns * MAX_HISTORY)
  std::vector<double> norm_hessian_force_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> norm_blocks_sensor_;     // (ns * ns) x MAX_HISTORY
  std::vector<double> norm_blocks_force_;      // (nv * nv) x MAX_HISTORY  

  // cost gradient
  std::vector<double> cost_gradient_prior_;    // nv * MAX_HISTORY
  std::vector<double> cost_gradient_sensor_;   // nv * MAX_HISTORY
  std::vector<double> cost_gradient_force_;    // nv * MAX_HISTORY

  // cost Hessian
  std::vector<double> cost_hessian_prior_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> cost_hessian_sensor_;    // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> cost_hessian_force_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> cost_hessian_band_;      // BandMatrixNonZeros(nv * MAX_HISTORY, 3 * nv)
  std::vector<double> cost_hessian_factor_;    // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)

  // cost scratch
  std::vector<double> scratch0_prior_;         // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> scratch1_prior_;         // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> scratch0_sensor_;        // (max(ns, 3 * nv) * max(ns, 3 * nv) * MAX_HISTORY)
  std::vector<double> scratch1_sensor_;        // (max(ns, 3 * nv) * max(ns, 3 * nv) * MAX_HISTORY)
  std::vector<double> scratch0_force_;         // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> scratch1_force_;         // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> scratch2_force_;         // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> scratch_prior_weight_;   // 2 * nv * nv

  // search direction
  std::vector<double> search_direction_;            // nv * MAX_HISTORY

  // status 
  bool hessian_factor_ = false;             // prior reset status
  int cost_count_;                          // number of cost evaluations
  int num_new_;                             // number of new trajectory elements
  bool initialized_ = false;                // flag for initialization

  // timers
  struct EstimatorTimers {
    double inverse_dynamics_derivatives;
    double velacc_derivatives;
    double jacobian_prior;
    double jacobian_sensor;
    double jacobian_force;
    double jacobian_total;
    double cost_prior_derivatives;
    double cost_sensor_derivatives;
    double cost_force_derivatives;
    double cost_total_derivatives;
    double cost_gradient;
    double cost_hessian;
    double cost_derivatives;
    double cost;
    double cost_prior;
    double cost_sensor;
    double cost_force;
    double cost_config_to_velacc;
    double cost_prediction;
    double residual_prior;
    double residual_sensor;
    double residual_force;
    double search_direction;
    double search;
    double configuration_update;
    double optimize;
    double prior_weight_update;
    double prior_set_weight;
    double update_trajectory;
    std::vector<double> prior_step;
    std::vector<double> sensor_step;
    std::vector<double> force_step;
  } timer_;

  // ----- GUI ----- //
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_ESTIMATOR_H_
