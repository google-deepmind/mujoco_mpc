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

#ifndef MJPC_TASKS_HUMANOID_BENCH_PUSH_PUSH_H_
#define MJPC_TASKS_HUMANOID_BENCH_PUSH_PUSH_H_

#include <memory>
#include <random>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class push : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const push *task)
        : mjpc::BaseResidualFn(task), task_(const_cast<push *>(task)) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;

   private:
    push *task_;
  };

  push() : residual_(this) {
    target_position_ = {0.85, 0.0, 1.0};
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis_x(0.7, 1.0);
    std::uniform_real_distribution<> dis_y(-0.5, 0.5);
    target_position_ = {dis_x(gen), dis_y(gen), 1.0};
  }

  void TransitionLocked(mjModel *model, mjData *data) override;

  void ResetLocked(const mjModel *model) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const

      override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual()

      override {
    return &residual_;
  }

 private:
  ResidualFn residual_;
  std::array<double, 3> target_position_;
};

class Push_H1 : public push {
 public:
  std::string Name() const override { return "Push H1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/push/Push_H1.xml");
  }
};

class G1_push : public push {
 public:
  std::string Name() const override { return "Push G1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/push/Push_G1.xml");
  }
};
}  // namespace mjpc

#endif  // MJPC_TASKS_HUMANOID_BENCH_PUSH_PUSH_H_
