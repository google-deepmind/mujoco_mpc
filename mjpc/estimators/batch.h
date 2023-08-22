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

#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/estimators/estimator.h"
#include "mjpc/estimators/gui.h"
#include "mjpc/estimators/trajectory.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// defaults
inline constexpr int kMinBatchHistory =
    3;  // minimum configuration trajectory length
inline constexpr int kMaxFilterHistory =
    32;  // maximum batch filter estimation horizon

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
inline constexpr double kMaxBatchRegularization = 1.0e12;
inline constexpr double kMinBatchRegularization = 1.0e-12;

// ----- batch estimator ----- //
// based on: "Physically-Consistent Sensor Fusion in Contact-Rich Behaviors"
class Batch : public Estimator {
 public:
  // constructor
  Batch() = default;
  Batch(int mode) {
    settings.filter = mode;
    settings.prior_flag = true;
    max_history_ = kMaxFilterHistory;
  }
  Batch(const mjModel* model, int length = 3, int max_history = 0) {
    // set max history length
    this->max_history_ = (max_history == 0 ? length : max_history);

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
  }

  // initialize
  void Initialize(const mjModel* model) override;

  // reset memory
  void Reset(const mjData* data = nullptr) override;

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
  int DimensionSensor() const override { return nsensordata_; };

  // set state
  void SetState(const double* state) override {
    // state
    mju_copy(this->state.data(), state, ndstate_);

    // -- configurations -- //
    int nq = model->nq;
    int t = 1;

    // q1
    configuration.Set(state, t);

    // q0
    double* q0 = configuration.Get(t - 1);
    mju_copy(q0, state, nq);
    mj_integratePos(model, q0, state + nq, -1.0 * model->opt.timestep);
  };

  // set time
  void SetTime(double time) override {
    // copy
    double time_copy = time;

    // t1
    times.Set(&time_copy, 1);

    // t0
    time_copy -= model->opt.timestep;
    times.Set(&time_copy, 0);

    // reset current time index
    current_time_index_ = 1;
  }

  // set covariance
  void SetCovariance(const double* covariance) override {
    mju_copy(this->covariance.data(), covariance, ndstate_ * ndstate_);
    // TODO(taylor): set prior weight = covariance^-1
  };

  // estimator-specific GUI elements
  void GUI(mjUI& ui, EstimatorGUIData& data) override;

  // set GUI data
  void SetGUIData(EstimatorGUIData& data) override;

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // set max history
  void SetMaxHistory(int length) { max_history_ = length; }

  // get max history
  int GetMaxHistory() { return max_history_; }

  // set configuration length
  void SetConfigurationLength(int length);

  // shift trajectory heads
  void Shift(int shift);

  // evaluate configurations
  void ConfigurationEvaluation(ThreadPool& pool);

  // compute total cost_
  double Cost(double* gradient, double* hessian, ThreadPool& pool);

  // optimize trajectory estimate
  void Optimize(ThreadPool& pool);

  // cost
  double GetCost() { return cost_; }
  double GetCostInitial() { return cost_initial_; }
  double GetCostPrior() { return cost_prior_; }
  double GetCostSensor() { return cost_sensor_; }
  double GetCostForce() { return cost_force_; }
  double* GetCostGradient() { return cost_gradient_.data(); }
  double* GetCostHessian() {
    // dimensions
    int nv = model->nv;
    int ntotal = nv * configuration_length_;
    int nband = 3 * nv;

    // resize
    cost_hessian_.resize(ntotal * ntotal);

    // band to dense
    mju_band2Dense(cost_hessian_.data(), cost_hessian_band_.data(), ntotal,
                   nband, 0, 1);

    // return dense Hessian
    return cost_hessian_.data();
  }
  double* GetCostHessianBand() { return cost_hessian_band_.data(); }

  // cost internals
  const double* GetResidualPrior() { return residual_prior_.data(); }
  const double* GetResidualSensor() { return residual_sensor_.data(); }
  const double* GetResidualForce() { return residual_force_.data(); }
  const double* GetJacobianPrior() {
    // dimensions
    int nv = model->nv;
    int ntotal = nv * configuration_length_;

    // resize
    jacobian_prior_.resize(ntotal * ntotal);

    // change setting
    int settings_cache = settings.assemble_prior_jacobian;
    settings.assemble_prior_jacobian = true;

    // loop over configurations to assemble Jacobian
    for (int t = 0; t < configuration_length_; t++) {
      BlockPrior(t);
    }

    // restore setting
    settings.assemble_prior_jacobian = settings_cache;

    // return dense Jacobian
    return jacobian_prior_.data();
  }
  const double* GetJacobianSensor() {
    // dimensions
    int nv = model->nv;
    int ntotal = nv * configuration_length_;
    int nsensortotal = nsensordata_ * (configuration_length_ - 1);

    // resize
    jacobian_sensor_.resize(nsensortotal * ntotal);

    // change setting
    int settings_cache = settings.assemble_sensor_jacobian;
    settings.assemble_sensor_jacobian = true;

    // loop over sensors
    for (int t = 0; t < configuration_length_ - 1; t++) {
      BlockSensor(t);
    }

    // restore setting
    settings.assemble_sensor_jacobian = settings_cache;

    // return dense Jacobian
    return jacobian_sensor_.data();
  }
  const double* GetJacobianForce() {
    // dimensions
    int nv = model->nv;
    int ntotal = nv * configuration_length_;
    int nforcetotal = nv * (configuration_length_ - 2);

    // resize
    jacobian_force_.resize(nforcetotal * ntotal);

    // change setting
    int settings_cache = settings.assemble_force_jacobian;
    settings.assemble_force_jacobian = true;

    // loop over sensors
    for (int t = 1; t < configuration_length_ - 1; t++) {
      BlockForce(t);
    }

    // restore setting
    settings.assemble_force_jacobian = settings_cache;

    // return dense Jacobian
    return jacobian_force_.data();
  }
  const double* GetNormGradientSensor() { return norm_gradient_sensor_.data(); }
  const double* GetNormGradientForce() { return norm_gradient_force_.data(); }
  const double* GetNormHessianSensor() {
    // dimensions
    int nsensortotal = nsensordata_ * (configuration_length_ - 1);

    // resize
    norm_hessian_sensor_.resize(nsensortotal * nsensortotal);

    // change setting
    int settings_cache = settings.assemble_sensor_norm_hessian;
    settings.assemble_sensor_norm_hessian = true;

    // evalute
    CostSensor(NULL, NULL);

    // restore setting
    settings.assemble_sensor_norm_hessian = settings_cache;

    // return dense Hessian
    return norm_hessian_sensor_.data();
  }
  const double* GetNormHessianForce() {
    // dimensions
    int nforcetotal = model->nv * (configuration_length_ - 2);

    // resize
    norm_hessian_force_.resize(nforcetotal * nforcetotal);

    // change setting
    int settings_cache = settings.assemble_force_norm_hessian;
    settings.assemble_force_norm_hessian = true;

    // evalute
    CostForce(NULL, NULL);

    // restore setting
    settings.assemble_force_norm_hessian = settings_cache;

    // return dense Hessian
    return norm_hessian_force_.data();
  }

  // get configuration length
  int ConfigurationLength() const { return configuration_length_; }

  // get number of sensors
  int NumberSensors() const { return nsensor_; }

  // measurement sensor start index
  int SensorStartIndex() const { return sensor_start_index_; }

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

  // set prior weights
  void SetPriorWeights(const double* weights, double scale = 1.0) {
    // dimension
    int nv = model->nv;
    int ntotal = nv * configuration_length_;
    int nband = 3 * nv;

    // allocate memory
    weight_prior_.resize(ntotal * ntotal);
    weight_prior_band_.resize(ntotal * (3 * nv));

    // set weights
    mju_copy(weight_prior_.data(), weights, ntotal * ntotal);

    // make block band
    DenseToBlockBand(weight_prior_.data(), ntotal, nv, 3);

    // dense to band
    mju_dense2Band(weight_prior_band_.data(), weight_prior_.data(), ntotal,
                   nband, 0);

    // set scaling
    scale_prior = scale;

    // set flag
    settings.prior_flag = true;
  }

  // get prior weights
  const double* PriorWeights() { return weight_prior_.data(); }

  // model
  mjModel* model = nullptr;

  // state (nstate_)
  std::vector<double> state;
  double time;

  // covariance (ndstate_ x ndstate_)
  std::vector<double> covariance;

  // process noise (ndstate_)
  std::vector<double> noise_process;

  // sensor noise (nsensor_)
  std::vector<double> noise_sensor;

  // prior
  double scale_prior;

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

  // norms
  std::vector<NormType> norm_type_sensor;  // num_sensor

  // norm parameters
  std::vector<double>
      norm_parameters_sensor;  // num_sensor x kMaxNormParameters

  // settings
  struct BatchSettings {
    bool prior_flag = false;  // flag for prior cost computation
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
    bool verbose_prior = false;  // flag for printing prior weight update status
    SearchType search_type =
        kCurveSearch;           // search type (line search, curve search)
    double step_scaling = 0.5;  // step size scaling
    double regularization_initial = 1.0e-12;       // initial regularization
    double regularization_scaling = mju_sqrt(10);  // regularization scaling
    bool time_scaling_force = true;              // scale force costs
    bool time_scaling_sensor = true;             // scale sensor costs
    double search_direction_tolerance = 1.0e-8;  // search direction tolerance
    double cost_tolerance = 1.0e-8;              // cost difference tolernace
    bool assemble_prior_jacobian = false;   // assemble dense prior Jacobian
    bool assemble_sensor_jacobian = false;  // assemble dense sensor Jacobian
    bool assemble_force_jacobian = false;   // assemble dense force Jacobian
    bool assemble_sensor_norm_hessian =
        false;  // assemble dense sensor norm Hessian
    bool assemble_force_norm_hessian =
        false;                            // assemble dense force norm Hessian
    bool filter = false;                  // filter mode
    bool recursive_prior_update = false;  // recursively update prior matrix
    bool joint_limits = true;  // enforce joint limits
  } settings;

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
  void TotalGradient(double* gradient);

  // compute total Hessian
  void TotalHessian(double* hessian);

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

  // update configuration trajectory
  void UpdateConfiguration(EstimatorTrajectory<double>& candidate,
                           const EstimatorTrajectory<double>& configuration,
                           const double* search_direction, double step_size);

  // initialize filter mode
  void InitializeFilter();

  // shift head and resize trajectories
  void ShiftResizeTrajectory(int new_head, int new_length);

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

  // enforce joint limits
  void JointLimits();

  // dimensions
  int nstate_;
  int ndstate_;
  int nsensordata_;
  int nsensor_;

  // sensor indexing
  int sensor_start_;
  int sensor_start_index_;

  // data
  std::vector<UniqueMjData> data_;

  // cost
  double cost_prior_;
  double cost_sensor_;
  double cost_force_;
  double cost_;
  double cost_initial_;
  double cost_previous_;

  // lengths
  int configuration_length_;  // T

  // configuration copy
  EstimatorTrajectory<double> configuration_copy_;  // nq x max_history_

  // residual
  std::vector<double> residual_prior_;   // nv x T
  std::vector<double> residual_sensor_;  // ns x (T - 1)
  std::vector<double> residual_force_;   // nv x (T - 2)

  // Jacobian
  std::vector<double> jacobian_prior_;   // (nv * T) * (nv * T)
  std::vector<double> jacobian_sensor_;  // (ns * (T - 1)) * (nv * T)
  std::vector<double> jacobian_force_;   // (nv * (T - 2)) * (nv * T)

  // prior Jacobian block
  EstimatorTrajectory<double>
      block_prior_current_configuration_;  // (nv * nv) x T

  // sensor Jacobian blocks (dqds, dvds, dads), (dsdq0, dsdq1, dsdq2)
  EstimatorTrajectory<double>
      block_sensor_configuration_;                     // (nsensordata * nv) x T
  EstimatorTrajectory<double> block_sensor_velocity_;  // (nsensordata * nv) x T
  EstimatorTrajectory<double>
      block_sensor_acceleration_;  // (nsensordata * nv) x T
  EstimatorTrajectory<double>
      block_sensor_configurationT_;  // (nv * nsensordata) x T
  EstimatorTrajectory<double>
      block_sensor_velocityT_;  // (nv * nsensordata) x T
  EstimatorTrajectory<double>
      block_sensor_accelerationT_;  // (nv * nsensordata) x T

  EstimatorTrajectory<double>
      block_sensor_previous_configuration_;  // (ns * nv) x T
  EstimatorTrajectory<double>
      block_sensor_current_configuration_;  // (ns * nv) x T
  EstimatorTrajectory<double>
      block_sensor_next_configuration_;  // (ns * nv) x T
  EstimatorTrajectory<double>
      block_sensor_configurations_;  // (ns * 3 * nv) x T

  EstimatorTrajectory<double> block_sensor_scratch_;  // max(nv, ns) x T

  // force Jacobian blocks (dqdf, dvdf, dadf), (dfdq0, dfdq1, dfdq2)
  EstimatorTrajectory<double> block_force_configuration_;  // (nv * nv) x T
  EstimatorTrajectory<double> block_force_velocity_;       // (nv * nv) x T
  EstimatorTrajectory<double> block_force_acceleration_;   // (nv * nv) x T

  EstimatorTrajectory<double>
      block_force_previous_configuration_;  // (nv * nv) x T
  EstimatorTrajectory<double>
      block_force_current_configuration_;                       // (nv * nv) x T
  EstimatorTrajectory<double> block_force_next_configuration_;  // (nv * nv) x T
  EstimatorTrajectory<double> block_force_configurations_;  // (nv * 3 * nv) x T

  EstimatorTrajectory<double> block_force_scratch_;  // (nv * nv) x T

  // velocity Jacobian blocks (dv1dq0, dv1dq1)
  EstimatorTrajectory<double>
      block_velocity_previous_configuration_;  // (nv * nv) x T
  EstimatorTrajectory<double>
      block_velocity_current_configuration_;  // (nv * nv) x T

  // acceleration Jacobian blocks (da1dq0, da1dq1, da1dq2)
  EstimatorTrajectory<double>
      block_acceleration_previous_configuration_;  // (nv * nv) x T
  EstimatorTrajectory<double>
      block_acceleration_current_configuration_;  // (nv * nv) x T
  EstimatorTrajectory<double>
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
  std::vector<double> cost_gradient_prior_;   // nv * max_history_
  std::vector<double> cost_gradient_sensor_;  // nv * max_history_
  std::vector<double> cost_gradient_force_;   // nv * max_history_
  std::vector<double> cost_gradient_;  // nv * max_history_

  // cost Hessian
  std::vector<double>
      cost_hessian_prior_band_;  // (nv * max_history_) * (3 * nv)
  std::vector<double>
      cost_hessian_sensor_band_;  // (nv * max_history_) * (3 * nv)
  std::vector<double>
      cost_hessian_force_band_;  // (nv * max_history_) * (3 * nv)
  std::vector<double>
      cost_hessian_;  // (nv * max_history_) * (nv * max_history_)
  std::vector<double> cost_hessian_band_;  // (nv * max_history_) * (3 * nv)
  std::vector<double>
      cost_hessian_band_factor_;  // (nv * max_history_) * (3 * nv)

  // cost scratch
  std::vector<double> scratch_prior_;  // nv * max_history_ + 12 * nv * nv
  std::vector<double>
      scratch_sensor_;  // 3 * nv + nsensor_data * 3 * nv + 9 * nv * nv
  std::vector<double> scratch_force_;     // 12 * nv * nv
  std::vector<double> scratch_expected_;  // nv * max_history_

  // search direction
  std::vector<double> search_direction_;  // nv * max_history_

  // prior weights
  std::vector<double>
      weight_prior_;  // (nv * max_history_) * (nv * max_history_)
  std::vector<double>
      weight_prior_band_;  // (nv * max_history_) * (nv * max_history_)

  // conditioned matrix
  std::vector<double> mat00_;
  std::vector<double> mat10_;
  std::vector<double> mat11_;
  std::vector<double> condmat_;
  std::vector<double> scratch0_condmat_;
  std::vector<double> scratch1_condmat_;

  // filter mode status
  int current_time_index_;

  // status (internal)
  int cost_count_;          // number of cost evaluations
  bool cost_skip_ = false;  // flag for only evaluating cost derivatives

  // status (external)
  int iterations_smoother_;       // total smoother iterations after Optimize
  int iterations_search_;         // total line search iterations
  double gradient_norm_;          // norm of cost gradient
  double regularization_;         // regularization
  double step_size_;              // step size for line search
  double search_direction_norm_;  // search direction norm
  BatchStatus solve_status_;      // estimator status
  double cost_difference_;        // cost difference: abs(cost - cost_previous)
  double improvement_;            // cost improvement
  double expected_;               // expected cost improvement
  double reduction_ratio_;  // reduction ratio: cost_improvement / expected cost
                            // improvement

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

  // max history
  int max_history_ = 3;

  // trajectory cache
  EstimatorTrajectory<double> configuration_cache_;           // nq x T
  EstimatorTrajectory<double> configuration_previous_cache_;  // nq x T
  EstimatorTrajectory<double> velocity_cache_;                // nv x T
  EstimatorTrajectory<double> acceleration_cache_;            // nv x T
  EstimatorTrajectory<double> act_cache_;                     // na x T
  EstimatorTrajectory<double> times_cache_;                   //  1 x T
  EstimatorTrajectory<double> ctrl_cache_;                    // nu x T
  EstimatorTrajectory<double> sensor_measurement_cache_;      // ns x T
  EstimatorTrajectory<double> sensor_prediction_cache_;       // ns x T
  EstimatorTrajectory<int> sensor_mask_cache_;                // num_sensor x T
  EstimatorTrajectory<double> force_measurement_cache_;       // nv x T
  EstimatorTrajectory<double> force_prediction_cache_;        // nv x T
};

// estimator status string
std::string StatusString(int code);

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_BATCH_H_
