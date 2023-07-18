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

#ifndef MJPC_ESTIMATORS_EKF_H_
#define MJPC_ESTIMATORS_EKF_H_

#include <mujoco/mujoco.h>

#include <mutex>
#include <vector>

#include "mjpc/utilities.h"

namespace mjpc {

// maximum terms
inline constexpr int kMaxProcessNoise = 1028;
inline constexpr int kMaxSensorNoise = 1028;

// https://stanford.edu/class/ee363/lectures/kf.pdf
class EKF {
 public:
  // constructor
  EKF() = default;
  EKF(const mjModel* model) {
    Initialize(model);
    Reset();
  }

  // initialize
  void Initialize(const mjModel* model);

  // reset memory
  void Reset();

  // update measurement
  void UpdateMeasurement(const double* ctrl, const double* sensor);

  // update time
  void UpdatePrediction();

  // dimension process 
  int DimensionProcess() const { return ndstate_; };

  // dimension sensor 
  int DimensionSensor() const { return nsensordata_; };

  // get measurement timer (ms)
  double TimerMeasurement() const { return timer_measurement_; };

  // get prediction timer (ms)
  double TimerPrediction() const { return timer_prediction_; };

  // estimator-specific GUI elements
  void GUI(mjUI& ui, double* process_noise, double* sensor_noise, double& timestep, int& integrator);

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift);

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

  // data
  mjData* data_;

  // correction (2nv + na)
  std::vector<double> correction_;

  // sensor Jacobian (nsensordata x (2nv + na))
  std::vector<double> sensor_jacobian_;

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

  // dynamics Jacobian ((2nv + na) x (2nv + na))
  std::vector<double> dynamics_jacobian_;

  // Kalman gain ((2nv + na) x nsensordata_)
  // TODO(taylor): unused..
  std::vector<double> kalman_gain_;

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

#endif  // MJPC_ESTIMATORS_EKF_H_
