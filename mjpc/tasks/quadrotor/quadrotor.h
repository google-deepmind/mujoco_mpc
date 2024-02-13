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

#ifndef MJPC_TASKS_QUADROTOR_QUADROTOR_H_
#define MJPC_TASKS_QUADROTOR_QUADROTOR_H_

#include <memory>
#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class Quadrotor : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    ResidualFn(const ResidualFn& residual) = default;
    explicit ResidualFn(const Quadrotor* task) : mjpc::BaseResidualFn(task) {}
    // --------------- Residuals for quadrotor task ---------------
    //   Number of residuals: 5
    //     Residual (0): position - goal position
    //     Residual (1): orientation - goal orientation
    //     Residual (2): linear velocity - goal linear velocity
    //     Residual (3): angular velocity - goal angular velocity
    //     Residual (4): control
    //   Number of parameters: 6
    // ------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };

  Quadrotor() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  int current_mode_ = 0;
  ResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_QUADROTOR_QUADROTOR_H_
