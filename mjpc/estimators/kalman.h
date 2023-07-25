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

#ifndef MJPC_ESTIMATORS_KALMAN_H_
#define MJPC_ESTIMATORS_KALMAN_H_

#include <mujoco/mujoco.h>

#include <mutex>
#include <vector>

#include "mjpc/estimators/estimator.h"
#include "mjpc/utilities.h"

namespace mjpc {

// https://stanford.edu/class/ee363/lectures/kf.pdf
class Kalman : public Estimator {
 public:
  // constructor
  Kalman() = default;
  Kalman(const mjModel* model) {
    Initialize(model);
    Reset();
  }

  // initialize
  void Initialize(const mjModel* model) override;

  // reset memory
  void Reset() override;

  // update measurement
  void UpdateMeasurement(const double* ctrl, const double* sensor);

  // update time
  void UpdatePrediction();

  // update 
  void Update(const double* ctrl, const double* sensor) override {
    UpdateMeasurement(ctrl, sensor);
    UpdatePrediction();
  }

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

  // get measurement timer (ms)
  double TimerMeasurement() const { return timer_measurement_; };

  // get prediction timer (ms)
  double TimerPrediction() const { return timer_prediction_; };

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

  // settings
  struct Settings {
    double epsilon = 1.0e-6;
    bool flg_centered = false;
    bool auto_timestep = false;
  } settings;

  // sensor start
  int sensor_start;
  int nsensor;

 private:
  // dimensions
  int nstate_;
  int ndstate_;
  int nsensordata_;

  // sensor start index
  int sensor_start_index_;

  // data
  mjData* data_;

  // correction (2nv + na)
  std::vector<double> correction_;

  // sensor Jacobian (nsensordata x (2nv + na))
  std::vector<double> sensor_jacobian_;

  // dynamics Jacobian ((2nv + na) x (2nv + na))
  std::vector<double> dynamics_jacobian_;

  // sensor error (nsensordata_)
  std::vector<double> sensor_error_;

  // timer (ms)
  double timer_measurement_;
  double timer_prediction_;

  // scratch
  std::vector<double> tmp0_;
  std::vector<double> tmp1_;
  std::vector<double> tmp2_;
  std::vector<double> tmp3_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_KALMAN_H_