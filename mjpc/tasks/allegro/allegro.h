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

#ifndef MJPC_TASKS_ALLEGRO_ALLEGRO_H_
#define MJPC_TASKS_ALLEGRO_ALLEGRO_H_

#include <mujoco/mujoco.h>

#include <memory>
#include <string>
#include <vector>

#include "mjpc/task.h"

namespace mjpc {
class Allegro : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const Allegro *task) : BaseResidualFn(task) {}
    void Residual(const mjModel *model, const mjData *data,
                  double *residual) const override;
  };
  Allegro() : residual_(this) {}

  // Reset the cube into the hand if it's on the floor
  void TransitionLocked(mjModel *model, mjData *data) override;
  void ModifyState(const mjModel *model, State *state) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn *InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  std::vector<double> pos_cube_ = std::vector<double>(3);
  std::vector<double> quat_cube_ = std::vector<double>(4);
  int rand1_ = 0;
  int rand2_ = 0;
  int rotation_counter = 0;
  std::chrono::steady_clock::time_point time_reset =
      std::chrono::steady_clock::now();  // time of last reset
};
}  // namespace mjpc

#endif  // MJPC_TASKS_ALLEGRO_ALLEGRO_H_
