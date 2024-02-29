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

#ifndef MJPC_ESTIMATORS_UNSCENTED_H_
#define MJPC_ESTIMATORS_UNSCENTED_H_

#include <mujoco/mujoco.h>

#include <mutex>
#include <vector>

#include "mjpc/estimators/estimator.h"
#include "mjpc/utilities.h"

namespace mjpc {

// Unscented Filtering and Nonlinear Estimation
// https://www.cs.ubc.ca/~murphyk/Papers/Julier_Uhlmann_mar04.pdf
class Unscented : public Estimator {
 public:
  // constructor
  Unscented() = default;
  Unscented(const mjModel* model) {
    Initialize(model);
    Reset();
  }

  // destructor
  ~Unscented() override {
    if (data_) mj_deleteData(data_);
    if (model) mj_deleteModel(model);
  }

  // initialize
  void Initialize(const mjModel* model) override;

  // reset memory
  void Reset(const mjData* data = nullptr) override;

  // compute sigma points
  void SigmaPoints();

  // evaluate sigma points
  void EvaluateSigmaPoints();

  // compute sigma point differences
  void SigmaPointDifferences();

  // compute sigma covariances
  void SigmaCovariances();

  // update
  void Update(const double* ctrl, const double* sensor, int mode = 0) override;

  // quaternion means
  void QuaternionMeans();

  // get state
  double* State() override { return state.data(); };

  // get covariance
  double* Covariance() override { return covariance.data(); };

  // get time
  double& Time() override { return time; };

  // get model
  mjModel* Model() override { return model; };

  // get data
  mjData* Data() override { return data_; };

  // get qfrc
  double* Qfrc() override { return data_->qfrc_actuator; };

  // get process noise
  double* ProcessNoise() override { return noise_process.data(); };

  // get sensor noise
  double* SensorNoise() override { return noise_sensor.data(); };

  // dimension process
  int DimensionProcess() const override { return ndstate_; };

  // dimension sensor
  int DimensionSensor() const override { return nsensordata_; };

  // set state
  void SetState(const double* state) override {
    mju_copy(this->state.data(), state, ndstate_);
  };

  // set time
  void SetTime(double time) override {
    this->time = time;
  }

  // set covariance
  void SetCovariance(const double* covariance) override {
    mju_copy(this->covariance.data(), covariance, ndstate_ * ndstate_);
  };

  // get update timer (ms)
  double TimerUpdate() const { return timer_update_; }

  // estimator-specific GUI elements
  void GUI(mjUI& ui) override;

  // set GUI data
  void SetGUIData() override;

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // model
  mjModel* model = nullptr;

  // state (nstate_)
  std::vector<double> state;
  double time;

  // covariance (ndstate_ x ndstate_)
  std::vector<double> covariance;

  // process noise (ndstate_)
  std::vector<double> noise_process;

  // sensor noise (nsensordata_)
  std::vector<double> noise_sensor;

  // sigma step
  double sigma_step;

  // weights
  double weight_mean0;
  double weight_covariance0;
  double weight_sigma;

  // settings
  struct Settings {
    double alpha = 1.0;
    double beta = 2.0;
  } settings;

 private:
  // dimensions
  int nstate_;
  int ndstate_;
  int nsensordata_;
  int nsensor_;
  int nsigma_;

  // sensor indexing
  int sensor_start_;
  int sensor_start_index_;

  // data
  mjData* data_ = nullptr;

  // correction (ndstate_)
  std::vector<double> correction_;

  // sensor error (nsensordata_)
  std::vector<double> sensor_error_;

  // sigma points (nstate x (2 * ndstate_ + 1))
  std::vector<double> sigma_;

  // states (nstate x (2 * ndstate_ + 1))
  std::vector<double> states_;

  // sensors (nsensordata_ x (2 * ndstate + 1))
  std::vector<double> sensors_;

  // state mean (nstate_)
  std::vector<double> state_mean_;

  // sensor mean (nsensordata_)
  std::vector<double> sensor_mean_;

  // covariance factor (ndstate_ x ndstate_)
  std::vector<double> covariance_factor_;

  // factor column (ndstate_)
  std::vector<double> factor_column_;

  // state difference (ndstate_ x nsigma_)
  std::vector<double> state_difference_;

  // sensor difference (nsensordata_ x nsigma_)
  std::vector<double> sensor_difference_;

  // covariance sensor (nsensordata_ x nsensordata_)
  std::vector<double> covariance_sensor_;

  // covariance state sensor (ndstate_ x nsensordata_)
  std::vector<double> covariance_state_sensor_;

  // covariance state state (ndstate_ x ndstate_)
  std::vector<double> covariance_state_state_;

  // sensor difference outer product
  std::vector<double> sensor_difference_outer_product_;

  // state sensor difference outer product
  std::vector<double> state_sensor_difference_outer_product_;

  // state state difference outer product
  std::vector<double> state_state_difference_outer_product_;

  // covariance sensor factor (nsensordata_ x nsensordata_)
  std::vector<double> covariance_sensor_factor_;

  // timer (ms)
  double timer_update_;

  // scratch
  std::vector<double> tmp0_;
  std::vector<double> tmp1_;

  // -- GUI data -- //

  // time step
  double gui_timestep_;

  // integrator
  int gui_integrator_;

  // process noise
  std::vector<double> gui_process_noise_;

  // sensor noise
  std::vector<double> gui_sensor_noise_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_UNSCENTED_H_
