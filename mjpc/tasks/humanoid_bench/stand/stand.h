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

#ifndef MJPC_TASKS_HUMANOID_BENCH_STAND_STAND_H_
#define MJPC_TASKS_HUMANOID_BENCH_STAND_STAND_H_

#include <memory>
#include <string>

#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mujoco/mujoco.h"

namespace mjpc {
class Stand : public Task {
 public:
  std::string Name() const override = 0;

  std::string XmlPath() const override = 0;

  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Stand *task) : mjpc::BaseResidualFn(task) {}

    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;
  };

  Stand() : residual_(this) {}

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }

  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};

class Stand_H1 : public Stand {
 public:
  std::string Name() const override { return "Stand H1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/stand/Stand_H1.xml");
  }
};

class Stand_G1 : public Stand {
 public:
  std::string Name() const override { return "Stand G1"; }

  std::string XmlPath() const override {
    return GetModelPath("humanoid_bench/stand/Stand_G1.xml");
  }
};
}  // namespace mjpc
#endif  // MJPC_TASKS_HUMANOID_BENCH_STAND_STAND_H_
