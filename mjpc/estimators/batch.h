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

#include <mutex>
#include <vector>

#include "mjpc/estimators/buffer.h"
#include "mjpc/estimators/include.h"
#include "mjpc/estimators/trajectory.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// ----- defaults ----- //
const int MIN_HISTORY = 3;      // minimum configuration trajectory length
const int max_history = 512;    // maximum configuration trajectory length

// batch estimator status 
enum BatchStatus : int {
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
const double MAX_REGULARIZATION = 1.0e12;
const double MIN_REGULARIZATION = 1.0e-12;

// ----- batch estimator ----- //
// based on: "Physically-Consistent Sensor Fusion in Contact-Rich Behaviors"
class Batch : public Estimator {
 public:
  // constructor
  Batch() = default;
  Batch(int mode) {
    settings.filter = mode;
  }
  Batch(const mjModel* model, int length=3, int max_history=0) {
    // set max history length
    this->max_history = (max_history == 0 ? length : max_history);

    // initialize memory
    Initialize(model);

    // set trajectory lengths
    SetConfigurationLength(length);

    // reset memory
    Reset();
  }

  // destructor
  ~Batch() {
    if (model) mj_deleteModel(model);
  };

  // initialize
  void Initialize(const mjModel* model) override;

  // reset memory
  void Reset() override;

  // update
  void Update(const double* ctrl, const double* sensor) override;

  // get state
  double* State() override { return state.data(); };

  // get covariance
  double* Covariance() override {
    // TODO(taylor): compute covariance from prior weight condition matrix
    return covariance.data();
  };

  // get time
  double& Time() override { return time; };

  // get model
  mjModel* Model() override { return model; };

  // get data
  mjData* Data() override { return data_[0].get(); };

  // get process noise 
  double* ProcessNoise() override { return noise_process.data(); };

  // get sensor noise 
  double* SensorNoise() override { return noise_sensor.data(); };

  // process dimension
  int DimensionProcess() const override { return ndstate_; };

  // sensor dimensino
  int DimensionSensor() const override { return nsensor; };

    // set state
  void SetState(const double* state) override {
    // state
    mju_copy(this->state.data(), state, ndstate_);

    // -- configurations -- //
    int nq = model->nq;
    int t = prediction_length_;

    // q1
    configuration.Set(state, t);

    // q0
    double* q0 = configuration.Get(t - 1);
    mju_copy(q0, state, nq);
    mj_integratePos(model, q0, state + nq, -1.0 * model->opt.timestep);
  };

  // set covariance
  void SetCovariance(const double* covariance) override {
    mju_copy(this->covariance.data(), covariance, ndstate_ * ndstate_);
    // TODO(taylor): set prior weight = covariance^-1
  };

  // estimator-specific GUI elements
  void GUI(mjUI& ui, double* process_noise, double* sensor_noise,
           double& timestep, int& integrator) override;

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // set configuration length
  void SetConfigurationLength(int length);

  // shift trajectory heads
  void Shift(int shift);

  // evaluate configurations
  void ConfigurationEvaluation(ThreadPool& pool);

  // compute total cost
  double Cost(double* gradient, double* hessian, ThreadPool& pool);

  // optimize trajectory estimate
  void Optimize(ThreadPool& pool);

  // cost internals
  const double* GetResidualPrior() { return residual_prior_.data(); }
  const double* GetResidualSensor() { return residual_sensor_.data(); }
  const double* GetResidualForce() { return residual_force_.data(); }
  const double* GetJacobianPrior() { return jacobian_prior_.data(); }
  const double* GetJacobianSensor() { return jacobian_sensor_.data(); }
  const double* GetJacobianForce() { return jacobian_force_.data(); }
  const double* GetNormGradientSensor() { return norm_gradient_sensor_.data(); }
  const double* GetNormGradientForce() { return norm_gradient_force_.data(); }
  const double* GetNormHessianSensor() { return norm_hessian_sensor_.data(); }
  const double* GetNormHessianForce() { return norm_hessian_force_.data(); }
  
  // get configuration length 
  int ConfigurationLength() const { return configuration_length_; }
  
  // get prediction length
  int PredictionLength() const { return prediction_length_; }

  // get number of sensors 
  int NumberSensors() const { return nsensor; }

  // get dimension of sensors 
  int SensorDimension() const { return nsensordata_; }

  // get status
  int IterationsSmoother() const { return iterations_smoother_; }
  int IterationsSearch() const { return iterations_search_; }
  double GradientNorm() const { return gradient_norm_; }
  double Regularization() const { return regularization_; }
  double StepSize() const { return step_size_; }
  double SearchDirectionNorm() const { return search_direction_norm_; }
  BatchStatus SolveStatus() const { return solve_status_; }
  double CostDifference() const { return cost_difference_; }
  double Improvement() const { return improvement_; } 
  double Expected() const { return expected_; }
  double ReductionRatio() const { return reduction_ratio_; }

  // model
  mjModel* model = nullptr;

  // state (nq + nv + na)
  std::vector<double> state;
  double time;

  // covariance
  std::vector<double> covariance;

  // process noise (2 * nv + na)
  std::vector<double> noise_process;

  // sensor noise (nsensor)
  std::vector<double> noise_sensor;

  // sensor start
  int sensor_start;
  int nsensor;

  // trajectories
  EstimatorTrajectory<double> configuration;           // nq x T
  EstimatorTrajectory<double> configuration_previous;  // nq x T
  EstimatorTrajectory<double> velocity;                // nv x T
  EstimatorTrajectory<double> acceleration;            // nv x T
  EstimatorTrajectory<double> act;                     // na x T
  EstimatorTrajectory<double> times;                   //  1 x T
  EstimatorTrajectory<double> ctrl;                    // nu x T
  EstimatorTrajectory<double> sensor_measurement;      // ns x T
  EstimatorTrajectory<double> sensor_prediction;       // ns x T
  EstimatorTrajectory<int> sensor_mask;                // num_sensor x T
  EstimatorTrajectory<double> force_measurement;       // nv x T
  EstimatorTrajectory<double> force_prediction;        // nv x T

  // cost
  double cost_prior;
  double cost_sensor;
  double cost_force;
  double cost;
  double cost_initial;
  double cost_previous;

  // cost gradient
  std::vector<double> cost_gradient;          // nv * max_history

  // cost Hessian
  std::vector<double> cost_hessian;           // (nv * max_history) * (nv * max_history)

  // prior weights
  std::vector<double> weight_prior;           // (nv * max_history) * (nv * max_history)
  
  // prior weights
  std::vector<double> weight_prior_band_;      // (nv * max_history) * (nv * max_history)

  // scale
  double scale_prior;

  // norms
  std::vector<NormType> norm_type_sensor;          // num_sensor

  // norm parameters
  std::vector<double> norm_parameters_sensor; // num_sensor x MAX_NORM_PARAMETERS

  // settings
  struct BatchSettings {
    bool prior_flag = true;                       // flag for prior cost computation
    bool sensor_flag = true;                      // flag for sensor cost computation
    bool force_flag = true;                       // flag for force cost computation
    int max_search_iterations = 1000;             // maximum number of line search iterations
    int max_smoother_iterations = 100;            // maximum number of smoothing iterations
    double gradient_tolerance = 1.0e-10;          // small gradient tolerance
    bool verbose_iteration = false;               // flag for printing optimize iteration
    bool verbose_optimize = false;                // flag for printing optimize status
    bool verbose_cost = false;                    // flag for printing cost
    bool verbose_prior = false;                   // flag for printing prior weight update status
    bool band_prior = true;                       // approximate covariance for prior
    SearchType search_type = kCurveSearch;        // search type (line search, curve search)
    double step_scaling = 0.5;                    // step size scaling
    double regularization_initial = 1.0e-12;      // initial regularization
    double regularization_scaling = mju_sqrt(10); // regularization scaling
    bool band_copy = true;                        // copy band matrices by block
    bool time_scaling = false;                    // scale sensor and force costs by time step
    double search_direction_tolerance = 1.0e-8;   // search direction tolerance
    double cost_tolerance = 1.0e-8;               // cost difference tolernace
    bool assemble_prior_jacobian = false;         // assemble dense prior Jacobian 
    bool assemble_sensor_jacobian = false;        // assemble dense sensor Jacobian
    bool assemble_force_jacobian = false;         // assemble dense force Jacobian 
    bool assemble_sensor_norm_hessian = false;    // assemble dense sensor norm Hessian 
    bool assemble_force_norm_hessian = false;     // assemble dense force norm Hessian
    bool filter = false;                          // filter mode
  } settings;
  
  // finite-difference settings
  struct FiniteDifferenceSettings {
    double tolerance = 1.0e-7;
    bool flg_actuation = 1;
  } finite_difference;

  // max history
  int max_history = 3;

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

  // prior Jacobian block
  void BlockPrior(int index);

  // prior Jacobian
  void JacobianPrior(ThreadPool& pool);

  // sensor residual
  void ResidualSensor();

  // sensor Jacobian blocks (dsdq0, dsdq1, dsdq2)
  void BlockSensor(int index);

  // sensor Jacobian
  void JacobianSensor(ThreadPool& pool);
  
  // force residual
  void ResidualForce();

  // force Jacobian blocks (dfdq0, dfdq1, dfdq2)
  void BlockForce(int index);

  // force Jacobian
  void JacobianForce(ThreadPool& pool);

  // search direction
  void SearchDirection();

//   // covariance 
//   void Covariance(ThreadPool& pool);

   // update configuration trajectory
  void UpdateConfiguration(EstimatorTrajectory<double>& candidate,
                           const EstimatorTrajectory<double>& configuration,
                           const double* search_direction, double step_size);

  // reset timers
  void ResetTimers();

  // print optimize iteration
  void PrintIteration();

  // print optimize status
  void PrintOptimize();

  // print cost
  void PrintCost();

  // increase regularization
  void IncreaseRegularization();

  // dimensions
  int nstate_;
  int ndstate_;
  int nsensordata_;

  // sensor start index
  int sensor_start_index_;

  // data
  std::vector<UniqueMjData> data_;

  // lengths
  int configuration_length_;                   // T
  int prediction_length_;                      // T - 2

  // configuration copy
  EstimatorTrajectory<double> configuration_copy_;  // nq x max_history

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
  EstimatorTrajectory<double> block_sensor_configuration_;            // (nsensordata * nv) x T
  EstimatorTrajectory<double> block_sensor_velocity_;                 // (nsensordata * nv) x T
  EstimatorTrajectory<double> block_sensor_acceleration_;             // (nsensordata * nv) x T
  EstimatorTrajectory<double> block_sensor_configurationT_;           // (nv * nsensordata) x T
  EstimatorTrajectory<double> block_sensor_velocityT_;                // (nv * nsensordata) x T
  EstimatorTrajectory<double> block_sensor_accelerationT_;            // (nv * nsensordata) x T

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

  // norm 
  std::vector<double> norm_sensor_;            // num_sensor * max_history 
  std::vector<double> norm_force_;             // nv * max_history

  // norm gradient
  std::vector<double> norm_gradient_sensor_;   // ns * max_history
  std::vector<double> norm_gradient_force_;    // nv * max_history

  // norm Hessian
  std::vector<double> norm_hessian_sensor_;    // (ns * ns * max_history)
  std::vector<double> norm_hessian_force_;     // (nv * max_history) * (nv * max_history)
  std::vector<double> norm_blocks_sensor_;     // (ns * ns) x max_history
  std::vector<double> norm_blocks_force_;      // (nv * nv) x max_history  

  // cost gradient
  std::vector<double> cost_gradient_prior_;    // nv * max_history
  std::vector<double> cost_gradient_sensor_;   // nv * max_history
  std::vector<double> cost_gradient_force_;    // nv * max_history

  // cost Hessian
  std::vector<double> cost_hessian_prior_;       // (nv * max_history) * (nv * max_history)
  std::vector<double> cost_hessian_sensor_;      // (nv * max_history) * (nv * max_history)
  std::vector<double> cost_hessian_force_;       // (nv * max_history) * (nv * max_history)
  std::vector<double> cost_hessian_band_;        // (nv * max_history) * (nv * max_history)
  std::vector<double> cost_hessian_band_factor_; // (nv * max_history) * (nv * max_history)
  std::vector<double> cost_hessian_factor_;      // (nv * max_history) * (nv * max_history)

  // cost scratch
  std::vector<double> scratch0_prior_;         // (nv * max_history) * (nv * max_history)
  std::vector<double> scratch1_prior_;         // (nv * max_history) * (nv * max_history)
  std::vector<double> scratch0_sensor_;        // (max(ns, 3 * nv) * max(ns, 3 * nv) * max_history)
  std::vector<double> scratch1_sensor_;        // (max(ns, 3 * nv) * max(ns, 3 * nv) * max_history)
  std::vector<double> scratch0_force_;         // (nv * max_history) * (nv * max_history)
  std::vector<double> scratch1_force_;         // (nv * max_history) * (nv * max_history)
  std::vector<double> scratch2_force_;         // (nv * max_history) * (nv * max_history)
  std::vector<double> scratch_prior_weight_;   // 2 * nv * nv
  std::vector<double> scratch_expected_;       // nv * max_history

  // search direction
  std::vector<double> search_direction_;       // nv * max_history

  // covariance 
  std::vector<double> prior_matrix_factor_;    // (nv * max_history) * (nv * max_history)
  std::vector<double> scratch0_covariance_;    // (nv * max_history) * (nv * max_history)
  std::vector<double> scratch1_covariance_;    // (nv * max_history) * (nv * max_history)

  // status (internal)
  int cost_count_;                          // number of cost evaluations
  bool cost_skip_ = false;                  // flag for only evaluating cost derivatives

  // status (external)
  int iterations_smoother_;                 // total smoother iterations after Optimize
  int iterations_search_;                   // total line search iterations
  double gradient_norm_;                    // norm of cost gradient
  double regularization_;                   // regularization
  double step_size_;                        // step size for line search
  double search_direction_norm_;            // search direction norm
  BatchStatus solve_status_;                // estimator status
  double cost_difference_;                  // cost difference: abs(cost - cost_previous)
  double improvement_;                      // cost improvement 
  double expected_;                         // expected cost improvement
  double reduction_ratio_;                  // reduction ratio: cost_improvement / expected cost improvement
  
  // timers
  struct BatchTimers {
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
    double update;
  } timer_;
};

// estimator status string
std::string StatusString(int code);

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_BATCH_H_
