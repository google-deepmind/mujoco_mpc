// Copyright 2022 DeepMind Technologies Limited
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

#ifndef MJPC_ESTIMATORS_GUI_H_
#define MJPC_ESTIMATORS_GUI_H_

#include <mujoco/mujoco.h>

#include <vector>

namespace mjpc {

// data that is modified by the GUI and later set in Estimator
class EstimatorGUIData {
 public:
  // constructor
  EstimatorGUIData() = default;

  // destructor
  ~EstimatorGUIData() = default;

  // Initialize
  void Initialize(const mjModel* model, int nprocess, int nsensor);

  // time step
  double timestep;

  // integrator
  int integrator;

  // process noise
  std::vector<double> process_noise;

  // sensor noise
  std::vector<double> sensor_noise;

  // estimation horizon
  int horizon;
};

}  // namespace mjpc

#endif  // MJPC_ESTIMATORS_GUI_H_
