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

// TODO(taylor): implement UKF algorithm

// THE SQUARE-ROOT UNSCENTED KALMAN FILTER FOR STATE AND PARAMETER-ESTIMATION
// https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.80.1421&rep=rep1&type=pdf
class Unscented : public Estimator {
 public:
  // constructor
  Unscented() = default;
  Unscented(const mjModel* model) {
    Initialize(model);
    Reset();
  }

  // initialize
  void Initialize(const mjModel* model) override;

  // reset memory
  void Reset() override;

  // compute sigma points 
  void SigmaPoints();

  // evaluate sigma points 
  void EvaluateSigmaPoints();

  // compute sigma point differences 
  void SigmaPointDifferences();

  // compute sigma covariances 
  void SigmaCovariances();

  // update
  void Update(const double* ctrl, const double* sensor) override;

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

  // get process noise 
  double* ProcessNoise() override { return noise_process.data(); };

  // get sensor noise 
  double* SensorNoise() override { return noise_sensor.data(); };

  // dimension process
  int DimensionProcess() const override { return ndstate_; };

  // dimension sensor
  int DimensionSensor() const override { return nsensordata_; };

  // get update timer (ms)
  double TimerUpdate() const { return timer_update_; };

  // estimator-specific GUI elements
  void GUI(mjUI& ui, double* process_noise, double* sensor_noise,
           double& timestep, int& integrator) override;

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // model
  mjModel* model;

  // state (nq + nv + na)
  std::vector<double> state;
  double time;

  // covariance
  std::vector<double> covariance;

  // process noise (2nv + na)
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

  // data
  mjData* data_;

  // correction (2nv + na)
  std::vector<double> correction_;

  // sensor start
  int sensor_start;
  int nsensor;

 private:
  // dimensions
  int nstate_;
  int ndstate_;
  int nsensordata_;
  int num_sigma_;

  // sensor start index
  int sensor_start_index_;

  // sensor error (nsensordata_)
  std::vector<double> sensor_error_;

  // sigma points (nstate x (2 * ndstate_ + 1))
  std::vector<double> sigma_;

  // states (nstate x (2 * ndstate_ + 1))
  std::vector<double> states_;

  // sensors (nsensordata_ x (2 * ndstate + 1))
  std::vector<double> sensors_;

  // state mean (nstate)
  std::vector<double> state_mean_;

  // sensor mean (nsensordata_)
  std::vector<double> sensor_mean_;

  // covariance factor (ndstate x ndstate)
  std::vector<double> covariance_factor_;

  // factor column (ndstate)
  std::vector<double> factor_column_;

  // state difference (ndstate x num_sigma)
  std::vector<double> state_difference_;

  // sensor difference (nsensordata_ x num_sigma)
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
  std::vector<double> tmp2_;
  std::vector<double> tmp3_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_UNSCENTED_H_