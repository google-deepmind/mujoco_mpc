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

#include <mujoco/mujoco.h>

#include <mutex>
#include <vector>

#include "mjpc/utilities.h"

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

// ground truth estimator
class GroundTruth : public Estimator {
  public:
  // constructor 
  GroundTruth() = default;
  GroundTruth(const mjModel* model) {
    Initialize(model);
    Reset();
  }

  // destructor 
  ~GroundTruth() {
    if (data_) mj_deleteData(data_);
    if (model) mj_deleteModel(model);
  }

  // initialize 
  void Initialize(const mjModel* model) override {
    // model
    if (this->model) mj_deleteModel(this->model);
    this->model = mj_copyModel(nullptr, model);

    // data
    data_ = mj_makeData(model);

    // -- dimensions -- //
    ndstate_ = 2 * model->nv + model->na;

    // sensor start index
    int sensor_start = GetNumberOrDefault(0, model, "estimator_sensor_start");

    // number of sensors
    int nsensor =
        GetNumberOrDefault(model->nsensor, model, "estimator_number_sensor");

    // sensor dimension
    nsensordata_ = 0;
    for (int i = 0; i < nsensor; i++) {
      nsensordata_ += model->sensor_dim[sensor_start + i];
    }

    // state 
    state.resize(model->nq + model->nv + model->na);

    // covariance 
    covariance.resize(ndstate_ * ndstate_);

    // process noise 
    noise_process.resize(ndstate_);

    // sensor noise 
    noise_sensor.resize(nsensordata_); // over allocate
  }

  // reset 
  void Reset() override {
    // dimensions 
    int nq = model->nq, nv = model->nv, na = model->na;
    int ndstate = 2 * nv + na;

    // set home keyframe
    int home_id = mj_name2id(model, mjOBJ_KEY, "home");
    if (home_id >= 0) mj_resetDataKeyframe(model, data_, home_id);

    // state
    mju_copy(state.data(), data_->qpos, nq);
    mju_copy(state.data() + nq, data_->qvel, nv);
    mju_copy(state.data() + nq + nv, data_->act, na);
    time = 0.0;

    // covariance
    mju_eye(covariance.data(), ndstate);
    double covariance_scl =
        GetNumberOrDefault(1.0e-4, model, "estimator_covariance_initial_scale");
    mju_scl(covariance.data(), covariance.data(), covariance_scl,
            ndstate * ndstate);

    // process noise
    double noise_process_scl =
        GetNumberOrDefault(1.0e-4, model, "estimator_process_noise_scale");
    std::fill(noise_process.begin(), noise_process.end(), noise_process_scl);

    // sensor noise
    double noise_sensor_scl =
        GetNumberOrDefault(1.0e-4, model, "estimator_sensor_noise_scale");
    std::fill(noise_sensor.begin(), noise_sensor.end(), noise_sensor_scl);
  }

  // update 
  void Update(const double* ctrl, const double* sensor) override {};

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

  // process dimension 
  int DimensionProcess() const override { return ndstate_; };

  // sensor dimension 
  int DimensionSensor() const override { return nsensordata_; };

  // set state
  void SetState(const double* state) override {
    mju_copy(this->state.data(), state, ndstate_);
  };

  // set covariance
  void SetCovariance(const double* covariance) override {
    mju_copy(this->covariance.data(), covariance, ndstate_ * ndstate_);
  };

  // estimator-specific GUI elements
  void GUI(mjUI& ui, double* process_noise, double* sensor_noise,
           double& timestep, int& integrator) override {};

  // estimator-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override {};

  // model
  mjModel* model = nullptr;

  // data 
  mjData* data_ = nullptr;

  // state (nq + nv + na)
  std::vector<double> state;
  double time;

  // covariance
  std::vector<double> covariance;

  // process noise 
  std::vector<double> noise_process;

  // sensor noise 
  std::vector<double> noise_sensor;

 private:
  // dimensions
  int ndstate_;
  int nsensordata_;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_ESTIMATOR_H_
