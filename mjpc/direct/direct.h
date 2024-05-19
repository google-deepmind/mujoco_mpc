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

#ifndef MJPC_DIRECT_DIRECT_H_
#define MJPC_DIRECT_DIRECT_H_

#include <memory>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/direct/model_parameters.h"
#include "mjpc/direct/trajectory.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// defaults
inline constexpr int kMinDirectHistory =
    3;  // minimum configuration trajectory length

// solve status flags
enum DirectStatus : int {
  kUnsolved = 0,
  kSearchFailure,
  kMaxIterationsFailure,
  kSmallDirectionFailure,
  kMaxRegularizationFailure,
  kCostDifferenceFailure,
  kExpectedDecreaseFailure,
  kSolved,
};

// search type for update
enum SearchType : int {
  kLineSearch = 0,
  kCurveSearch,
  kNumSearch,
};

// maximum / minimum regularization
inline constexpr double kMaxDirectRegularization = 1.0e12;
inline constexpr double kMinDirectRegularization = 1.0e-12;

// ----- direct optimization with MuJoCo inverse dynamics ----- //
class Direct {
 public:
  // constructor
  explicit Direct(int num_threads = NumAvailableHardwareThreads())
      : model_parameters_(LoadModelParameters()), pool_(num_threads) {}

  // constructor
  explicit Direct(const mjModel* model, int length = 3, int max_history = 0);

  // destructor
  virtual ~Direct() {
    if (model) mj_deleteModel(model);
  }

  // initialize
  void Initialize(const mjModel* model);

  // reset memory
  void Reset(const mjData* data = nullptr);

  // set max history
  void SetMaxHistory(int length) { max_history_ = length; }

  // get max history
  int GetMaxHistory() { return max_history_; }

  // set configuration length
  void SetConfigurationLength(int length);

  // evaluate configurations
  void ConfigurationEvaluation();

  // compute total cost_
  // virtual function so derived classes can add most cost terms
  virtual double Cost(double* gradient, double* hessian);

  // optimize trajectory estimate
  void Optimize();

  // cost
  double GetCost() { return cost_; }
  double GetCostInitial() { return cost_initial_; }
  double GetCostSensor() { return cost_sensor_; }
  double GetCostForce() { return cost_force_; }
  double GetCostParameter() { return cost_parameter_; }
  double* GetCostGradient() { return cost_gradient_.data(); }
  double* GetCostHessian();
  double* GetCostHessianBand() { return cost_hessian_band_.data(); }

  // cost internals
  const double* GetResidualSensor() { return residual_sensor_.data(); }
  const double* GetResidualForce() { return residual_force_.data(); }
  const double* GetJacobianSensor();
  const double* GetJacobianForce();
  const double* GetNormGradientSensor() { return norm_gradient_sensor_.data(); }
  const double* GetNormGradientForce() { return norm_gradient_force_.data(); }
  const double* GetNormHessianSensor();
  const double* GetNormHessianForce();

  // get configuration length
  int ConfigurationLength() const { return configuration_length_; }

  // get number of sensors
  int NumberSensors() const { return nsensor_; }

  // sensor dimension
  int DimensionSensor() const { return nsensordata_; }

  // measurement sensor start index
  int SensorStartIndex() const { return sensor_start_index_; }

  // get number of parameters
  int NumberParameters() const { return nparam_; }

  // get status
  int IterationsSmoother() const { return iterations_smoother_; }
  int IterationsSearch() const { return iterations_search_; }
  double GradientNorm() const { return gradient_norm_; }
  double Regularization() const { return regularization_; }
  double StepSize() const { return step_size_; }
  double SearchDirectionNorm() const { return search_direction_norm_; }
  DirectStatus SolveStatus() const { return solve_status_; }
  double CostDifference() const { return cost_difference_; }
  double Improvement() const { return improvement_; }
  double Expected() const { return expected_; }
  double ReductionRatio() const { return reduction_ratio_; }

  // model
  mjModel* model = nullptr;

  // process noise (ndstate_)
  std::vector<double> noise_process;

  // sensor noise (nsensor_)
  std::vector<double> noise_sensor;

  // trajectories
  DirectTrajectory<double> configuration;           // nq x T
  DirectTrajectory<double> configuration_previous;  // nq x T
  DirectTrajectory<double> velocity;                // nv x T
  DirectTrajectory<double> acceleration;            // nv x T
  DirectTrajectory<double> act;                     // na x T
  DirectTrajectory<double> times;                   //  1 x T
  DirectTrajectory<double> sensor_measurement;      // ns x T
  DirectTrajectory<double> sensor_prediction;       // ns x T
  DirectTrajectory<int> sensor_mask;                // num_sensor x T
  DirectTrajectory<double> force_measurement;       // nv x T
  DirectTrajectory<double> force_prediction;        // nv x T

  // parameters
  std::vector<double> parameters;           // nparam
  std::vector<double> parameters_previous;  // nparam
  std::vector<double> noise_parameter;      // nparam

  // norms
  std::vector<NormType> norm_type_sensor;  // num_sensor

  // norm parameters
  std::vector<double>
      norm_parameters_sensor;  // num_sensor x kMaxNormParameters

  // settings
  struct DirectSettings {
    bool sensor_flag = true;  // flag for sensor cost computation
    bool force_flag = true;   // flag for force cost computation
    int max_search_iterations =
        1000;  // maximum number of line search iterations
    int max_smoother_iterations =
        100;  // maximum number of smoothing iterations
    double gradient_tolerance = 1.0e-10;  // small gradient tolerance
    bool verbose_iteration = false;  // flag for printing optimize iteration
    bool verbose_optimize = false;   // flag for printing optimize status
    bool verbose_cost = false;       // flag for printing cost
    SearchType search_type =
        kCurveSearch;           // search type (line search, curve search)
    double step_scaling = 0.5;  // step size scaling
    double regularization_initial = 1.0e-12;       // initial regularization
    double regularization_scaling = mju_sqrt(10);  // regularization scaling
    bool time_scaling_force = true;                // scale force costs
    bool time_scaling_sensor = true;               // scale sensor costs
    double search_direction_tolerance = 1.0e-8;    // search direction tolerance
    double cost_tolerance = 1.0e-8;                // cost difference tolernace
    bool assemble_sensor_jacobian = false;  // assemble dense sensor Jacobian
    bool assemble_force_jacobian = false;   // assemble dense force Jacobian
    bool assemble_sensor_norm_hessian =
        false;  // assemble dense sensor norm Hessian
    bool assemble_force_norm_hessian =
        false;  // assemble dense force norm Hessian
    bool first_step_position_sensors =
        true;  // evaluate position sensors at first time step
    bool last_step_position_sensors =
        false;  // evaluate position sensors at last time step
    bool last_step_velocity_sensors =
        false;  // evaluate velocity sensors at last time step
    bool assemble_cost_hessian = false;  // assemble dense cost Hessian
  } settings;

  // finite-difference settings
  struct FiniteDifferenceSettings {
    double tolerance = 1.0e-7;
    bool flg_actuation = 1;
  } finite_difference;

 protected:
  // convert sequence of configurations to velocities, accelerations
  void ConfigurationToVelocityAcceleration();

  // compute sensor and force predictions via inverse dynamics
  void InverseDynamicsPrediction();

  // compute finite-difference velocity, acceleration derivatives
  void VelocityAccelerationDerivatives();

  // compute inverse dynamics derivatives (via finite difference)
  void InverseDynamicsDerivatives();

  // evaluate configurations derivatives
  void ConfigurationDerivative();

  // ----- sensor ----- //
  // cost
  double CostSensor(double* gradient, double* hessian);

  // residual
  void ResidualSensor();

  // Jacobian blocks (dsdq0, dsdq1, dsdq2)
  void BlockSensor(int index);

  // Jacobian
  void JacobianSensor();

  // ----- force ----- //
  // cost
  double CostForce(double* gradient, double* hessian);

  // residual
  void ResidualForce();

  // Jacobian blocks (dfdq0, dfdq1, dfdq2)
  void BlockForce(int index);

  // Jacobian
  void JacobianForce();

  // compute total gradient
  void TotalGradient(double* gradient);

  // compute total Hessian
  void TotalHessian(double* hessian);

  // search direction, returns false if regularization maxes out
  bool SearchDirection();

  // update configuration trajectory
  void UpdateConfiguration(DirectTrajectory<double>& candidate,
                           const DirectTrajectory<double>& configuration,
                           const double* search_direction, double step_size);

  // reset timers
  void ResetTimers();

  // print optimize status
  void PrintOptimize();

  // print cost
  void PrintCost();

  // increase regularization
  void IncreaseRegularization();

  // derivatives of force and sensor model wrt parameters
  void ParameterJacobian(int index);

  // dimensions
  int nstate_ = 0;
  int ndstate_ = 0;
  int nsensordata_ = 0;
  int nsensor_ = 0;

  int ntotal_ = 0;  // total number of decision variable
  int nvel_ = 0;    // number of configuration (derivatives) variables
  int nparam_ = 0;  // number of parameter variable (ndense)
  int nband_ = 0;   // cost Hessian band dimension

  // sensor indexing
  int sensor_start_ = 0;
  int sensor_start_index_ = 0;

  // perturbed models (for parameter estimation)
  std::vector<UniqueMjModel> model_perturb_;

  // data
  std::vector<UniqueMjData> data_;

  // cost
  double cost_sensor_ = 0.0;
  double cost_force_ = 0.0;
  double cost_parameter_ = 0.0;
  double cost_ = 0.0;
  double cost_initial_ = 0.0;
  double cost_previous_ = 0.0;

  // lengths
  int configuration_length_ = 0;  // T

  // configuration copy
  DirectTrajectory<double> configuration_copy_;  // nq x max_history_

  // residual
  std::vector<double> residual_sensor_;  // ns x (T - 1)
  std::vector<double> residual_force_;   // nv x (T - 2)

  // Jacobian
  std::vector<double> jacobian_sensor_;  // (ns * (T - 1)) * (nv * T + nparam)
  std::vector<double> jacobian_force_;   // (nv * (T - 2)) * (nv * T + nparam)

  // sensor Jacobian blocks (dqds, dvds, dads), (dsdq0, dsdq1, dsdq2)
  DirectTrajectory<double>
      block_sensor_configuration_;                  // (nsensordata * nv) x T
  DirectTrajectory<double> block_sensor_velocity_;  // (nsensordata * nv) x T
  DirectTrajectory<double>
      block_sensor_acceleration_;  // (nsensordata * nv) x T
  DirectTrajectory<double>
      block_sensor_configurationT_;                  // (nv * nsensordata) x T
  DirectTrajectory<double> block_sensor_velocityT_;  // (nv * nsensordata) x T
  DirectTrajectory<double>
      block_sensor_accelerationT_;  // (nv * nsensordata) x T

  DirectTrajectory<double>
      block_sensor_previous_configuration_;  // (ns * nv) x T
  DirectTrajectory<double>
      block_sensor_current_configuration_;                    // (ns * nv) x T
  DirectTrajectory<double> block_sensor_next_configuration_;  // (ns * nv) x T
  DirectTrajectory<double> block_sensor_configurations_;  // (ns * 3 * nv) x T

  DirectTrajectory<double> block_sensor_scratch_;  // max(nv, ns) x T

  // force Jacobian blocks (dqdf, dvdf, dadf), (dfdq0, dfdq1, dfdq2)
  DirectTrajectory<double> block_force_configuration_;  // (nv * nv) x T
  DirectTrajectory<double> block_force_velocity_;       // (nv * nv) x T
  DirectTrajectory<double> block_force_acceleration_;   // (nv * nv) x T

  DirectTrajectory<double>
      block_force_previous_configuration_;                      // (nv * nv) x T
  DirectTrajectory<double> block_force_current_configuration_;  // (nv * nv) x T
  DirectTrajectory<double> block_force_next_configuration_;     // (nv * nv) x T
  DirectTrajectory<double> block_force_configurations_;  // (nv * 3 * nv) x T

  DirectTrajectory<double> block_force_scratch_;  // (nv * nv) x T

  // sensor Jacobian blocks wrt parameters (dsdp, dpds)
  DirectTrajectory<double>
      block_sensor_parameters_;  // (nsensordata * nparam_) x T
  DirectTrajectory<double>
      block_sensor_parametersT_;  // (nparam_ * nsensordata) x T

  // force Jacobian blocks wrt parameters (dpdf)
  DirectTrajectory<double> block_force_parameters_;  // (nparam_ * nv) x T

  // velocity Jacobian blocks (dv1dq0, dv1dq1)
  DirectTrajectory<double>
      block_velocity_previous_configuration_;  // (nv * nv) x T
  DirectTrajectory<double>
      block_velocity_current_configuration_;  // (nv * nv) x T

  // acceleration Jacobian blocks (da1dq0, da1dq1, da1dq2)
  DirectTrajectory<double>
      block_acceleration_previous_configuration_;  // (nv * nv) x T
  DirectTrajectory<double>
      block_acceleration_current_configuration_;  // (nv * nv) x T
  DirectTrajectory<double>
      block_acceleration_next_configuration_;  // (nv * nv) x T

  // norm
  std::vector<double> norm_sensor_;  // num_sensor * max_history_
  std::vector<double> norm_force_;   // nv * max_history_

  // norm gradient
  std::vector<double> norm_gradient_sensor_;  // ns * max_history_
  std::vector<double> norm_gradient_force_;   // nv * max_history_

  // norm Hessian
  std::vector<double> norm_hessian_sensor_;  // (ns * ns * max_history_)
  std::vector<double>
      norm_hessian_force_;  // (nv * max_history_) * (nv * max_history_)
  std::vector<double> norm_blocks_sensor_;  // (ns * ns) x max_history_
  std::vector<double> norm_blocks_force_;   // (nv * nv) x max_history_

  // cost gradient
  std::vector<double> cost_gradient_sensor_;  // nv * max_history_ + nparam
  std::vector<double> cost_gradient_force_;   // nv * max_history_ + nparam
  std::vector<double> cost_gradient_;         // nv * max_history_ + nparam

  // cost Hessian
  std::vector<double>
      cost_hessian_sensor_band_;  // (nv * max_history_) * (3 * nv) + nparam *
                                  // (nv * max_history_)
  std::vector<double>
      cost_hessian_force_band_;  // (nv * max_history_) * (3 * nv) + nparam *
                                 // (nv * max_history_)
  std::vector<double> cost_hessian_;  // (nv * max_history_ + nparam) * (nv *
                                      // max_history_ + nparam)
  std::vector<double> cost_hessian_band_;  // (nv * max_history_) * (3 * nv) +
                                           // nparam * (nv * max_history_)
  std::vector<double>
      cost_hessian_band_factor_;  // (nv * max_history_) * (3 * nv) + nparam *
                                  // (nv * max_history_)

  // cost scratch
  std::vector<double>
      scratch_sensor_;  // 3 * nv + nsensor_data * 3 * nv + 9 * nv * nv
  std::vector<double> scratch_force_;  // 12 * nv * nv
  std::vector<double>
      scratch_expected_;  // nv * max_history_ + nparam * (nv * max_history_)

  // search direction
  std::vector<double>
      search_direction_;  // nv * max_history_ + nparam * (nv * max_history_)

  // parameters copy
  std::vector<double> parameters_copy_;  // nparam x T

  // dense cost Hessian rows (for parameter derivatives)
  std::vector<double> dense_force_parameter_;   // nparam x ntotal
  std::vector<double> dense_sensor_parameter_;  // nparam x ntotal
  std::vector<double> dense_parameter_;         // nparam x ntotal

  // model parameters
  std::vector<std::unique_ptr<mjpc::ModelParameters>> model_parameters_;
  int model_parameters_id_ = 0;

  // status (internal)
  int cost_count_ = 0;      // number of cost evaluations
  bool cost_skip_ = false;  // flag for only evaluating cost derivatives

  // status (external)
  int iterations_smoother_ = 0;  // total smoother iterations after Optimize
  int iterations_search_ = 0;    // total line search iterations
  double gradient_norm_ = 0.0;   // norm of cost gradient
  double regularization_ = 0.0;  // regularization
  double step_size_ = 0.0;       // step size for line search
  double search_direction_norm_ = 0.0;     // search direction norm
  DirectStatus solve_status_ = kUnsolved;  // solve status
  double cost_difference_ = 0.0;  // cost difference: abs(cost - cost_previous)
  double improvement_ = 0.0;      // cost improvement
  double expected_ = 0.0;         // expected cost improvement
  double reduction_ratio_ = 0.0;  // reduction ratio: cost_improvement /
                                  // expected cost improvement

  // timers
  struct DirectTimers {
    double inverse_dynamics_derivatives = 0.0;
    double velacc_derivatives = 0.0;
    double jacobian_sensor = 0.0;
    double jacobian_force = 0.0;
    double jacobian_total = 0.0;
    double cost_sensor_derivatives = 0.0;
    double cost_force_derivatives = 0.0;
    double cost_total_derivatives = 0.0;
    double cost_gradient = 0.0;
    double cost_hessian = 0.0;
    double cost_derivatives = 0.0;
    double cost = 0.0;
    double cost_sensor = 0.0;
    double cost_force = 0.0;
    double cost_config_to_velacc = 0.0;
    double cost_prediction = 0.0;
    double residual_sensor = 0.0;
    double residual_force = 0.0;
    double search_direction = 0.0;
    double search = 0.0;
    double configuration_update = 0.0;
    double optimize = 0.0;
    double update_trajectory = 0.0;
    std::vector<double> sensor_step;
    std::vector<double> force_step;
    double update = 0.0;
  } timer_;

  // max history
  int max_history_ = 3;

  // threadpool
  ThreadPool pool_;
};

// optimizer status string
std::string StatusString(int code);

}  // namespace mjpc

#endif  // MJPC_DIRECT_DIRECT_H_
