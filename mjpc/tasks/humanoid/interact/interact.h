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

#ifndef MJPC_TASKS_HUMANOID_INTERACT_INTERACT_H_
#define MJPC_TASKS_HUMANOID_INTERACT_INTERACT_H_

#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include "mjpc/task.h"
#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"
#include "mjpc/tasks/humanoid/interact/motion_strategy.h"

namespace mjpc::humanoid {

// ---------- Constants ----------------- //
constexpr int kHeadHeightParameterIndex = 0;
constexpr int kTorsoHeightParameterIndex = 1;
constexpr int kNumberOfFreeJoints = 0;

// ---------- Enums ----------------- //
enum TaskMode : int {
  kSitting = 0,
  kStanding = 1,
  kRelaxing = 2,
  kStayingStill = 3
};

// ----------- Default weights for the residual terms ----------------- //
const std::vector<std::vector<double>> default_weights = {
    {10, 10, 5, 5, 0, 20, 30, 0, 0, 0, 0.01, .1, 80.},    // to sit
    {10, 0, 1, 1, 80, 0, 0, 100, 0, 0, 0.01, 0.025, 0.},  // to stand
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.01, .8, 80.},        // to relax
    {0, 0, 0, 0, 0, 0, 0, 0, 0, 50, 20, .025, 80.},       // to stay still
};

// ----------- Default colors for the contact pair points ------------ //
constexpr float kContactPairColor[kNumberOfContactPairsInteract][4] = {
    {0., 0., 1., 0.8},  // blue
    {0., 1., 0., 0.8},  // green
    {0., 1., 1., 0.8},  // cyan
    {1., 0., 0., 0.8},  // red
    {1., 0., 1., 0.8},  // magenta
};
constexpr float kFacingDirectionColor[] = {1., 1., 1., 0.8};
constexpr double kVisualPointSize[3] = {0.02};

class Interact : public Task {
 public:
  class ResidualFn : public mjpc::BaseResidualFn {
   public:
    explicit ResidualFn(const Interact* task,
                        const ContactKeyframe& kf = ContactKeyframe(),
                        int current_mode = kSitting)
        : mjpc::BaseResidualFn(task),
          residual_keyframe_(kf),
          current_task_mode_((TaskMode)current_mode) {}

    // ------------------ Residuals for interaction task ------------
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;

   protected:
    ContactKeyframe residual_keyframe_;

   private:
    friend class Interact;

    TaskMode current_task_mode_;

    void UpResidual(const mjModel* model, const mjData* data, double* residual,
                    std::string&& name, int* counter) const;

    void HeadHeightResidual(const mjModel* model, const mjData* data,
                            double* residual, int* counter) const;

    void TorsoHeightResidual(const mjModel* model, const mjData* data,
                             double* residual, int* counter) const;

    void KneeFeetXYResidual(const mjModel* model, const mjData* data,
                            double* residual, int* counter) const;

    void COMFeetXYResidual(const mjModel* model, const mjData* data,
                           double* residual, int* counter) const;

    void TorsoTargetResidual(const mjModel* model, const mjData* data,
                             double* residual, int* counter) const;

    void FacingDirectionResidual(const mjModel* model, const mjData* data,
                                 double* residual, int* counter) const;

    void ContactResidual(const mjModel* model, const mjData* data,
                         double* residual, int* counter) const;
  };

  Interact() : residual_(this) {}

  void TransitionLocked(mjModel* model, mjData* data) override;

  std::string Name() const override;
  std::string XmlPath() const override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this, residual_.residual_keyframe_,
                                        residual_.current_task_mode_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
  MotionStrategy motion_strategy_;

  // draw task-related geometry in the scene
  void ModifyScene(const mjModel* model, const mjData* data,
                   mjvScene* scene) const override;
};

}  // namespace mjpc::humanoid

#endif  // MJPC_TASKS_HUMANOID_INTERACT_INTERACT_H_
