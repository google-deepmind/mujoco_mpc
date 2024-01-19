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

#ifndef MJPC_TASKS_CUBE_SOLVE_H_
#define MJPC_TASKS_CUBE_SOLVE_H_

#include <string>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
class CubeSolve : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;
  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const CubeSolve* task, int current_mode = 0)
        : BaseResidualFn(task), current_mode_(current_mode) {
          mju_zero(goal_, 6);
        }
    ResidualFn(const ResidualFn&) = default;
    // ---------- Residuals for cube solving manipulation task ----
    //   Number of residuals:
    // ------------------------------------------------------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   private:
    friend class CubeSolve;
    int current_mode_ = 0;
    double goal_[6] = {0};
  };
  CubeSolve() : residual_(this) {
    // path to transition model xml
    std::string path = GetModelPath("cube/transition_model.xml");

    // load transition model
    constexpr int kErrorLength = 1024;
    char load_error[kErrorLength] = "";
    transition_model_ =
        mj_loadXML(path.c_str(), nullptr, load_error, kErrorLength);
    transition_data_ = mj_makeData(transition_model_);
  }

  // ----- Transition for cube solving manipulation task -----
  //   If cube is within tolerance or floor ->
  //   reset cube into hand.
  // ---------------------------------------------------------
  void TransitionLocked(mjModel* model, mjData* data) override;
  
  // modes
  enum CubeSolveMode {
    kModeWait= 0,
    kModeManual,
    kModeScramble,
    kModeSolve,
  };

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  friend class ResidualFn;
  ResidualFn residual_;
  mjModel* transition_model_;
  mjData* transition_data_;
  int num_scramble_ = 2;
  std::vector<int> face_;
  std::vector<int> direction_;
  std::vector<double> goal_cache_;
  int goal_index_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_CUBE_SOLVE_H_
