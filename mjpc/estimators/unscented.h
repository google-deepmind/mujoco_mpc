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

#endif  // MJPC_ESTIMATORS_UNSCENTED_H_
