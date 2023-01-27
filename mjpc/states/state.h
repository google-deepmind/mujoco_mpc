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

#ifndef MJPC_STATES_STATE_H_
#define MJPC_STATES_STATE_H_

#include <shared_mutex>
#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc {

// data and methods for state
class State {
 public:
  friend class StateTest;

  // constructor
  State() = default;

  // destructor
  ~State() = default;

  // ----- methods ----- //

  // initialize settings
  void Initialize(const mjModel* model) {}

  // allocate memory
  void Allocate(const mjModel* model);

  // reset memory to zeros
  void Reset();

  // set state from data
  void Set(const mjModel* model, const mjData* data);

  // copy into destination
  void CopyTo(double* dst_state, double* dst_mocap, double* dst_userdata, double* time) const;

  const std::vector<double>& state() const { return state_; }
  const std::vector<double>& mocap() const { return mocap_; }
  const std::vector<double>& userdata() const { return userdata_; }
  double time() const { return time_; }

 private:
  std::vector<double> state_;  // (state dimension x 1)
  std::vector<double> mocap_;  // (mocap dimension x 1)
  std::vector<double> userdata_; // (nuserdata x 1)
  double time_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_STATES_STATE_H_
