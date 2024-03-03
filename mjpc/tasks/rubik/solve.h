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

#ifndef MJPC_TASKS_RUBIK_SOLVE_H_
#define MJPC_TASKS_RUBIK_SOLVE_H_

#include <algorithm>
#include <memory>
#include <vector>
#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {

class Rubik : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const Rubik* task, int current_mode = 0,
                        int goal_index = 0)
        : BaseResidualFn(task),
          current_mode_(current_mode),
          goal_index_(goal_index) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class Rubik;
    int current_mode_ = 0;
    int goal_index_ = 0;
  };

  Rubik();
  ~Rubik();

  void TransitionLocked(mjModel* model, mjData* data) override;

  // modes
  enum RubikMode {
    kModeScramble = 0,
    kModeSolve,
    kModeWait,
    kModeManual,
  };

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.current_mode_,
                                        residual_.goal_index_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  mjModel* transition_model_ = nullptr;
  mjData* transition_data_ = nullptr;
  std::vector<int> face_;
  std::vector<int> direction_;
  std::vector<double> goal_cache_;
  int goal_index_;
};

}  // namespace mjpc

#endif  // MJPC_TASKS_RUBIK_SOLVE_H_
