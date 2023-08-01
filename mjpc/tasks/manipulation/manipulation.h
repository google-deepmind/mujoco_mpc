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

#ifndef MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_
#define MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_


#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/tasks/manipulation/common.h"

namespace mjpc::manipulation {
class Bring : public ThreadSafeTask {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Bring* task, ModelValues values)
        : mjpc::BaseResidualFn(task), model_vals_(std::move(values)) {}

    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
   private:
    friend class Bring;
    ModelValues model_vals_;
  };

  Bring() : residual_(this, ModelValues()) {}
  void TransitionLocked(mjModel* model, mjData* data,
                        std::mutex* mutex) override;
  void ResetLocked(const mjModel* model) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.model_vals_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc::manipulation


#endif  // MJPC_MJPC_TASKS_MANIPULATION_MANIPULATION_H_
