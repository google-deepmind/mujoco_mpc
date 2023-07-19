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

// https://stanford.edu/class/ee363/lectures/kf.pdf
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
};

// ground truth
class GroundTruth : public Estimator {
  public:
  // constructor 
  GroundTruth() = default;
  GroundTruth(const mjModel* model) {
    Initialize(model);
    Reset();
  }

  // destructor 
  ~GroundTruth() = default;

  // initialize 
  void Initialize(const mjModel* model) override {
    // model
    if (this->model) mj_deleteModel(this->model);
    this->model = mj_copyModel(nullptr, model);

    // data
    data_ = mj_makeData(model);

    // state 
    state.resize(model->nq + model->nv + model->na);

    // covariance 
    int ndstate = 2 * model->nv + model->na;
    covariance.resize(ndstate * ndstate);

    // process noise 
    noise_process.resize(ndstate);

    // sensor noise 
    noise_sensor.resize(model->nsensordata); // over allocate
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
        GetNumberOrDefault(1.0e-5, model, "estimator_covariance_initial_scale");
    mju_scl(covariance.data(), covariance.data(), covariance_scl,
            ndstate * ndstate);

    // process noise 
    std::fill(noise_process.begin(), noise_process.end(), 0.0);

    // sensor noise 
    std::fill(noise_sensor.begin(), noise_sensor.end(), 0.0);
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

  // model
  mjModel* model;

  // data 
  mjData* data_;

  // state (nq + nv + na)
  std::vector<double> state;
  double time;

  // covariance
  std::vector<double> covariance;

  // process noise 
  std::vector<double> noise_process;

  // sensor noise 
  std::vector<double> noise_sensor;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_ESTIMATOR_H_
