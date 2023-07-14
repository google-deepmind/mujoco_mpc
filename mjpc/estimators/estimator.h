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
const int MAX_HISTORY = 128;  // maximum configuration trajectory length

const int NUM_FORCE_TERMS = 3;
const int MAX_NORM_PARAMETERS = 3;

// search type for update
enum SearchType : int {
  kLineSearch = 0,
  kCurveSearch,
  kNumSearch,
};

// maximum / minimum regularization
const double MAX_REGULARIZATION = 1.0e6;
const double MIN_REGULARIZATION = 1.0e-6;

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
  void Initialize(mjModel* model);  // TODO(taylor): const

  // set configuration length
  void SetConfigurationLength(int length);

  // shift trajectory heads
  void ShiftTrajectoryHead(int shift);

  // reset memory
  void Reset();

  // prior cost
  double CostPrior(double* gradient, double* hessian);

  // prior residual
  void ResidualPrior();

  // set block in prior Jacobian
  void SetBlockPrior(int index);

  // prior Jacobian block
  void BlockPrior(int index);

  // prior Jacobian
  void JacobianPrior(ThreadPool& pool);

  // sensor cost
  double CostSensor(double* gradient, double* hessian);

  // sensor residual
  void ResidualSensor();

  // set block in sensor Jacobian
  void SetBlockSensor(int index);

  // sensor Jacobian blocks (dsdq0, dsdq1, dsdq2)
  void BlockSensor(int index);

  // sensor Jacobian
  void JacobianSensor(ThreadPool& pool);

  // force cost
  double CostForce(double* gradient, double* hessian);

  // force residual
  void ResidualForce();

  // set block in force Jacobian
  void SetBlockForce(int index);

  // force Jacobian blocks (dfdq0, dfdq1, dfdq2)
  void BlockForce(int index);

  // force Jacobian
  void JacobianForce(ThreadPool& pool);

  // compute sensor and force predictions via inverse dynamics
  void InverseDynamicsPrediction(ThreadPool& pool);

  // compute inverse dynamics derivatives (via finite difference)
  void InverseDynamicsDerivatives(ThreadPool& pool);

  // update configuration trajectory
  void UpdateConfiguration(EstimatorTrajectory<double>& candidate,
                           const EstimatorTrajectory<double>& configuration,
                           const double* search_direction, double step_size);

  // convert sequence of configurations to velocities, accelerations
  void ConfigurationToVelocityAcceleration();

  // compute finite-difference velocity, acceleration derivatives
  void VelocityAccelerationDerivatives();

  // compute total cost
  double Cost(ThreadPool& pool);

  // compute total gradient
  void CostGradient();

  // compute total Hessian
  void CostHessian();

  // prior weight update
  void PriorWeightUpdate(ThreadPool& pool);

  // optimize trajectory estimate
  void Optimize(ThreadPool& pool);

  // regularize Hessian
  void Regularize();

  // search direction
  void SearchDirection();

  // print optimize status
  void PrintOptimize();

  // print cost
  void PrintCost();

  // print update prior weight status
  void PrintPriorWeightUpdate();

  // reset timers
  void ResetTimers();

  // get qpos estimate
  double* GetPosition();

  // get qvel estimate
  double* GetVelocity();

  // initialize trajectories
  void InitializeTrajectories(const EstimatorTrajectory<double>& measurement,
                              const EstimatorTrajectory<int>& measurement_mask,
                              const EstimatorTrajectory<double>& ctrl,
                              const EstimatorTrajectory<double>& time);

  // update trajectories
  int UpdateTrajectories(const EstimatorTrajectory<double>& measurement,
                         const EstimatorTrajectory<int>& measurement_mask,
                         const EstimatorTrajectory<double>& ctrl,
                         const EstimatorTrajectory<double>& time);

  // update
  void Update(const Buffer& buffer, ThreadPool& pool);

  // get terms from GUI
  void GetGUI();

  // set terms to GUI
  void SetGUI();

  // model
  mjModel* model_;

  // data
  std::vector<UniqueMjData> data_;

  // trajectories
  int configuration_length_;                   // T
  int prediction_length_;                      // T - 2
  EstimatorTrajectory<double> configuration_;  // nq x T
  EstimatorTrajectory<double> velocity_;       // nv x T
  EstimatorTrajectory<double> acceleration_;   // nv x T
  EstimatorTrajectory<double> time_;           //  1 x T

  // prior
  EstimatorTrajectory<double> configuration_prior_;  // nq x T

  // sensor
  int dim_sensor_;                                   // ns
  int num_sensor_;                                   // num_sensor
  int num_free_;
  std::vector<bool> free_dof_;                       // flag indicating free joint dof
  EstimatorTrajectory<double> sensor_measurement_;   // ns x T
  EstimatorTrajectory<double> sensor_prediction_;    // ns x T
  EstimatorTrajectory<int> sensor_mask_;             // num_sensor x T

  // forces
  EstimatorTrajectory<double> force_measurement_;    // nv x T
  EstimatorTrajectory<double> force_prediction_;     // nv x T

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

  // cost
  double cost_prior_;
  double cost_sensor_;
  double cost_force_;
  double cost_;
  double cost_initial_;

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

  // prior weights
  std::vector<double> weight_prior_dense_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> weight_prior_band_;      // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> scratch_prior_weight_;   // 2 * nv * nv

  double scale_prior_;

  // sensor scale
  std::vector<double> scale_sensor_;           // num_sensor

  // force scale (free, ball, slide, hinge)
  std::vector<double> scale_force_;            // NUM_FORCE_TERMS

  // cost norms
  std::vector<NormType> norm_sensor_;          // num_sensor
  NormType norm_force_[NUM_FORCE_TERMS];       // NUM_FORCE_TERMS

  // cost norm parameters
  std::vector<double> norm_parameters_sensor_; // num_sensor x MAX_NORM_PARAMETERS
  std::vector<double> norm_parameters_force_;  // NUM_FORCE_TERMS x MAX_NORM_PARAMETERS

  // norm gradient
  std::vector<double> norm_gradient_sensor_;   // ns * MAX_HISTORY
  std::vector<double> norm_gradient_force_;    // nv * MAX_HISTORY

  // norm Hessian
  std::vector<double> norm_hessian_sensor_;    // (ns * ns * MAX_HISTORY)
  std::vector<double> norm_hessian_force_;     // (nv * MAX_HISTORY) * (nv * MAX_HISTORY)
  std::vector<double> norm_blocks_sensor_;     // (ns * ns) x MAX_HISTORY
  std::vector<double> norm_blocks_force_;      // (nv * nv) x MAX_HISTORY

  // candidate
  EstimatorTrajectory<double> configuration_copy_;  // nq x MAX_HISTORY

  // search direction
  std::vector<double> search_direction_;            // nv * MAX_HISTORY

  // regularization
  double regularization_;

  // search type
  SearchType search_type_;
  double step_size_;

  // timing
  double timer_total_;
  double timer_inverse_dynamics_derivatives_;
  double timer_velacc_derivatives_;
  double timer_jacobian_prior_;
  double timer_jacobian_sensor_;
  double timer_jacobian_force_;
  double timer_jacobian_total_;
  double timer_cost_prior_derivatives_;
  double timer_cost_sensor_derivatives_;
  double timer_cost_force_derivatives_;
  double timer_cost_total_derivatives_;
  double timer_cost_gradient_;
  double timer_cost_hessian_;
  double timer_cost_derivatives_;
  double timer_cost_;
  double timer_cost_prior_;
  double timer_cost_sensor_;
  double timer_cost_force_;
  double timer_cost_config_to_velacc_;
  double timer_cost_prediction_;
  double timer_residual_prior_;
  double timer_residual_sensor_;
  double timer_residual_force_;
  double timer_search_direction_;
  double timer_search_;
  double timer_configuration_update_;
  double timer_optimize_;
  double timer_prior_weight_update_;
  double timer_prior_set_weight_;
  double timer_update_trajectory_;

  std::vector<double> timer_prior_step_;
  std::vector<double> timer_sensor_step_;
  std::vector<double> timer_force_step_;

  // cost flags
  bool prior_flag_ = true;
  bool sensor_flag_ = true;
  bool force_flag_ = true;

  // state index
  int state_index_;

  // status
  int iterations_smoother_;                 // total smoother iterations after Optimize
  int iterations_line_search_;              // total line search iterations
  bool hessian_factor_ = false;             // prior reset status
  int cost_count_;
  int num_new_;                             // number of new elements
  double gradient_norm_;                    // norm of cost gradient

  bool initialized_ = false;                // flag for initialization

  // settings
  int max_line_search_ = 10;                // maximum number of line search iterations
  int max_smoother_iterations_ = 10;        // maximum number of smoothing iterations
  double gradient_tolerance_ = 1.0e-5;      // small gradient tolerance
  bool verbose_optimize_ = false;           // flag for printing optimize status
  bool verbose_cost_ = false;               // flag for printing cost
  bool verbose_prior_ = false;              // flag for printing prior weight update status
  bool band_covariance_ = true;             // approximate covariance for prior
  double step_scaling_ = 0.5;               // step size scaling
  double regularization_initial_ = 1.0e-5;  // initial regularization
  double regularization_scaling_ = 10.0;    // regularization scaling
  bool band_copy_ = true;                   // copy band matrices by block
  bool reuse_data_ = false;                 // flag for resuing data previously computed
  bool skip_update_prior_weight = false;    // flag for skipping update prior weight
  bool update_prior_weight_ = true;         // flag for updating prior weights
  bool time_scaling_ = false;               // scale sensor and force costs by time step

  // finite-difference settings
  struct FiniteDifferenceSettings {
    double tolerance = 1.0e-5;
    bool flg_actuation = 0;
  } finite_difference_;

  // ----- GUI terms ----- //

  // TODO(taylor): initialize all properly

  // settings
  int gui_configuration_length_;
  int gui_max_smoother_iterations_;

  // weights
  double gui_scale_prior_;
  std::vector<double> gui_weight_sensor_;
  std::vector<double> gui_weight_force_;

  // costs
  double gui_cost_prior_;
  double gui_cost_sensor_;
  double gui_cost_force_;
  double gui_cost_;

  // status
  double gui_regularization_;
  double gui_step_size_;

  // timers

  // mutex
  std::mutex mutex_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_ESTIMATOR_H_
