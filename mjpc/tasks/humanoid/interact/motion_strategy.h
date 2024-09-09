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

#ifndef MJPC_TASKS_HUMANOID_INTERACT_MOTION_STRATEGY_H_
#define MJPC_TASKS_HUMANOID_INTERACT_MOTION_STRATEGY_H_

#include <mujoco/mujoco.h>

#include <string>
#include <vector>

#include "mjpc/tasks/humanoid/interact/contact_keyframe.h"

namespace mjpc::humanoid {

/*
This class holds the motion strategy, e.g. given a sequence of keyframes, it
manages the initial and current state and any changes to the sequence.
*/
class MotionStrategy {
 public:
  MotionStrategy() = default;
  ~MotionStrategy() = default;

  explicit MotionStrategy(const std::vector<ContactKeyframe>& keyframes)
      : contact_keyframes_(keyframes), current_keyframe_index_(0) {}

  void Reset();
  void Clear();

  bool HasKeyframes() const { return !contact_keyframes_.empty(); }
  ContactKeyframe& GetCurrentKeyframe() {
    return contact_keyframes_[current_keyframe_index_];
  }
  const ContactKeyframe& GetCurrentKeyframe() const {
    return contact_keyframes_[current_keyframe_index_];
  }
  const int GetCurrentKeyframeIndex() const { return current_keyframe_index_; }
  const std::vector<ContactKeyframe>& GetContactKeyframes() const {
    return contact_keyframes_;
  }
  const int GetKeyframesCount() const { return contact_keyframes_.size(); }
  const mjtNum GetCurrentKeyframeStartTime() const {
    return current_keyframe_start_time_;
  }
  void SetCurrentKeyframeStartTime(const mjtNum start_time) {
    current_keyframe_start_time_ = start_time;
  }
  const mjtNum GetCurrentKeyframeSuccessStartTime() const {
    return current_keyframe_success_start_time_;
  }
  void SetCurrentKeyframeSuccessStartTime(const mjtNum start_time) {
    current_keyframe_success_start_time_ = start_time;
  }
  void UpdateCurrentKeyframe(const int index) {
    current_keyframe_index_ = index;
  }
  void SetContactKeyframes(const std::vector<ContactKeyframe>& keyframes) {
    contact_keyframes_ = keyframes;
  }

  int NextKeyframe();
  void ClearKeyframes();

  double CalculateTotalKeyframeDistance(
      const mjData* data, const ContactKeyframeErrorType error_type =
                              ContactKeyframeErrorType::kNorm) const;

 private:
  std::vector<ContactKeyframe> contact_keyframes_;
  int current_keyframe_index_;
  mjtNum current_keyframe_start_time_;
  mjtNum current_keyframe_success_start_time_;
};

}  // namespace mjpc::humanoid

#endif  // MJPC_TASKS_HUMANOID_INTERACT_MOTION_STRATEGY_H_
