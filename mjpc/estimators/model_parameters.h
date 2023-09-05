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

#ifndef MJPC_ESTIMATORS_MODEL_PARAMETERS_H_
#define MJPC_ESTIMATORS_MODEL_PARAMETERS_H_

#include <memory>
#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc {

// virtual class for setting model parameters
class ModelParameters {
 public:
  // destructor
  virtual ~ModelParameters() = default;

  // set parameters
  virtual void Set(mjModel* model, const double* parameters, int dim) = 0;
};

// model parameter class for 1D particle w/ damping
class Particle1DParameters : public ModelParameters {
 public:
  // constructor
  Particle1DParameters() = default;

  // destructor
  ~Particle1DParameters() = default;

  // set parameters
  void Set(mjModel* model, const double* parameters, int dim) override {
    // set damping value
    model->dof_damping[0] = parameters[0];
  }
};

// Loads all available ModelParameters
std::vector<std::unique_ptr<mjpc::ModelParameters>> LoadModelParameters();

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_MODEL_PARAMETERS_H_
