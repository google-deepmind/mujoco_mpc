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

#ifndef MJPC_TASKS_OP3_STAND_H_
#define MJPC_TASKS_OP3_STAND_H_

#include <memory>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {

class OP3 : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const OP3* task, int current_mode = 0)
        : BaseResidualFn(task), current_mode_(current_mode) {}
    // ------- Residuals for OP3 task ------------
    //     Residual(0): height - feet height
    //     Residual(1): balance
    //     Residual(2): center of mass xy velocity
    //     Residual(3): ctrl - ctrl_nominal
    //     Residual(4): upright
    //     Residual(5): joint velocity
    // -------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class OP3;
    int current_mode_;

    // modes
    enum OP3Mode {
      kModeStand = 0,
      kModeHandstand,
    };
  };

  OP3() : residual_(this) {}
  void TransitionLocked(mjModel* model, mjData* data) override;

  // default height goals
  constexpr static double kModeHeight[2] = {0.38, 0.57};

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_OP3_STAND_H_
