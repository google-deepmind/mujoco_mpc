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

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/estimators/estimator.h"
#include "mjpc/direct/direct.h"
#include "mjpc/direct/trajectory.h"
#include "mjpc/norm.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc {

// max filter history
inline constexpr int kMaxFilterHistory = 64;

// ----- batch estimator ----- //
// based on: "Physically-Consistent Sensor Fusion in Contact-Rich Behaviors"
class Batch : public Direct, public Estimator {
 public:
  // constructor
  explicit Batch(int num_threads = 1) : Direct(num_threads) {
    max_history_ = kMaxFilterHistory;
  }
  Batch(const mjModel* model, int length = 3,
        int max_history = kMaxFilterHistory);

  // destructor
  ~Batch() override = default;

  // total cost
  double Cost(double* gradient, double* hessian) override;

  // initialize
  void Initialize(const mjModel* model) override;

  // reset memory
  void Reset(const mjData* data = nullptr) override;

  // update
  void Update(const double* ctrl, const double* sensor, int mode = 0) override;

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

  // get qfrc
  double* Qfrc() override {
    return force_prediction.Get(configuration_length_ - 2);
  };

  // get process noise
  double* ProcessNoise() override { return noise_process.data(); };

  // get sensor noise
  double* SensorNoise() override { return noise_sensor.data(); };

  // process dimension
  int DimensionProcess() const override { return ndstate_; };

  // sensor dimension
  int DimensionSensor() const override { return nsensordata_; };

  // set state
  void SetState(const double* state) override;

  // set time
  void SetTime(double time) override;

  // set covariance
  void SetCovariance(const double* covariance) override {
    mju_copy(this->covariance.data(), covariance, ndstate_ * ndstate_);
    // TODO(taylor): set prior weight = covariance^-1
  };

  // estimator-specific GUI elements
  void GUI(mjUI& ui) override;

  // set GUI data
  void SetGUIData() override;

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // set max history
  void SetMaxHistory(int length) { max_history_ = length; }

  // get max history
  int GetMaxHistory() { return max_history_; }

  // shift trajectory heads
  void Shift(int shift);

  // cost
  double GetCostPrior() { return cost_prior_; }

  // cost internals
  const double* GetResidualPrior() { return residual_prior_.data(); }
  const double* GetJacobianPrior();

  // set prior weights
  void SetPriorWeights(const double* weights, double scale = 1.0);

  // get prior weights
  const double* PriorWeights() { return weight_prior_.data(); }

  // state (nstate_)
  std::vector<double> state;
  double time;

  // covariance (ndstate_ x ndstate_)
  std::vector<double> covariance;

  // prior
  double scale_prior;

  // settings
  struct FilterSettings {
    bool verbose_prior = false;  // flag for printing prior weight update status
    bool assemble_prior_jacobian = false;   // assemble dense prior Jacobian
    bool recursive_prior_update = false;  // recursively update prior matrix
  } filter_settings;

 private:
  // ----- prior ----- //
  // cost
  double CostPrior(double* gradient, double* hessian);

  // residual
  void ResidualPrior();

  // Jacobian block
  void BlockPrior(int index);

  // Jacobian
  void JacobianPrior();

  // initialize filter mode
  void InitializeFilter();

  // shift head and resize trajectories
  void ShiftResizeTrajectory(int new_head, int new_length);

  // reset direct and prior timers
  void ResetTimers();

  // cost
  double cost_prior_;

  // residual
  std::vector<double> residual_prior_;   // nv x T

  // Jacobian
  std::vector<double> jacobian_prior_;   // (nv * T) * (nv * T + nparam)

  // prior Jacobian block
  DirectTrajectory<double>
      block_prior_current_configuration_;  // (nv * nv) x T

  // cost gradient
  std::vector<double> cost_gradient_prior_;   // nv * max_history_ + nparam

  // cost Hessian
  std::vector<double>
      cost_hessian_prior_band_;  // (nv * max_history_) * (3 * nv) + nparam *
                                 // (nv * max_history_)

  // cost scratch
  std::vector<double> scratch_prior_;  // nv * max_history_ + 12 * nv * nv +
                                       // nparam * (nv * max_history_)

  // prior weights
  std::vector<double> weight_prior_;  // (nv * max_history_ + nparam) * (nv *
                                      // max_history_ + nparam)
  std::vector<double> weight_prior_band_;  // (nv * max_history_ + nparam) * (nv
                                           // * max_history_ + nparam)

  // conditioned matrix
  std::vector<double> mat00_;
  std::vector<double> mat10_;
  std::vector<double> mat11_;
  std::vector<double> condmat_;
  std::vector<double> scratch0_condmat_;
  std::vector<double> scratch1_condmat_;

  // filter mode status
  int current_time_index_;

  // timers
  struct FilterTimers {
    double cost_prior_derivatives;
    double cost_prior;
    double residual_prior;
    double jacobian_prior;
    double prior_weight_update;
    double prior_set_weight;
    std::vector<double> prior_step;
  } filter_timer_;

  // trajectory cache
  DirectTrajectory<double> configuration_cache_;           // nq x T
  DirectTrajectory<double> configuration_previous_cache_;  // nq x T
  DirectTrajectory<double> velocity_cache_;                // nv x T
  DirectTrajectory<double> acceleration_cache_;            // nv x T
  DirectTrajectory<double> act_cache_;                     // na x T
  DirectTrajectory<double> times_cache_;                   //  1 x T
  DirectTrajectory<double> sensor_measurement_cache_;      // ns x T
  DirectTrajectory<double> sensor_prediction_cache_;       // ns x T
  DirectTrajectory<int> sensor_mask_cache_;                // num_sensor x T
  DirectTrajectory<double> force_measurement_cache_;       // nv x T
  DirectTrajectory<double> force_prediction_cache_;        // nv x T

  // -- GUI data -- //

  // time step
  double gui_timestep_;

  // integrator
  int gui_integrator_;

  // process noise
  std::vector<double> gui_process_noise_;

  // sensor noise
  std::vector<double> gui_sensor_noise_;

  // scale prior
  double gui_scale_prior_;

  // estimation horizon
  int gui_horizon_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_BATCH_H_
