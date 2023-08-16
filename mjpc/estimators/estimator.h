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
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc {

// maximum terms
inline constexpr int kMaxProcessNoise = 1028;
inline constexpr int kMaxSensorNoise = 1028;

// virtual estimator class
class Estimator {
 public:
  // destructor
  virtual ~Estimator() = default;

  // initialize
  virtual void Initialize(const mjModel* model) = 0;

  // reset memory
  virtual void Reset() = 0;

  // update
  virtual void Update(const double* ctrl, const double* sensor) = 0;

  // get state
  virtual double* State() = 0;

  // get covariance
  virtual double* Covariance() = 0;

  // get time
  virtual double& Time() = 0;

  // get model
  virtual mjModel* Model() = 0;

  // get data
  virtual mjData* Data() = 0;

  // process noise
  virtual double* ProcessNoise() = 0;

  // sensor noise
  virtual double* SensorNoise() = 0;

  // process dimension
  virtual int DimensionProcess() const = 0;

  // sensor dimension
  virtual int DimensionSensor() const = 0;

  // set state
  virtual void SetState(const double* state) = 0;

  // set covariance
  virtual void SetCovariance(const double* covariance) = 0;

  // estimator-specific GUI elements
  virtual void GUI(mjUI& ui, double* process_noise, double* sensor_noise,
                   double& timestep, int& integrator) = 0;

  // estimator-specific plots
  virtual void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                     int planner_shift, int timer_shift, int planning,
                     int* shift) = 0;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_ESTIMATOR_H_
