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

#include "mjpc/tasks/humanoid/interact/motion_strategy.h"

#include "mjpc/utilities.h"

namespace mjpc::humanoid {

void MotionStrategy::Clear() {
  contact_keyframes_.clear();
  contact_keyframes_[current_keyframe_index_].Reset();
  current_keyframe_index_ = 0;
}

void MotionStrategy::Reset() { current_keyframe_index_ = 0; }

int MotionStrategy::NextKeyframe() {
  current_keyframe_index_ =
      (current_keyframe_index_ + 1) % contact_keyframes_.size();
  return current_keyframe_index_;
}

void MotionStrategy::ClearKeyframes() {
  contact_keyframes_.clear();
  current_keyframe_index_ = 0;
}

double MotionStrategy::CalculateTotalKeyframeDistance(
    const mjData* data, const ContactKeyframeErrorType error_type) const {
  double error[kNumberOfContactPairsInteract] = {0.};

  // iterate through all pairs
  for (int i = 0; i < kNumberOfContactPairsInteract; i++) {
    const ContactPair& contact =
        contact_keyframes_[current_keyframe_index_].contact_pairs[i];
    if (contact.body1 != kNotSelectedInteract &&
        contact.body2 != kNotSelectedInteract) {
      double dist[3] = {0.};
      contact.GetDistance(dist, data);
      error[i] = mju_norm3(dist);
    }
  }

  double total_error = 0.;
  switch (error_type) {
    case ContactKeyframeErrorType::kMax:
      total_error =
          *std::max_element(error, error + kNumberOfContactPairsInteract);
      break;
    case ContactKeyframeErrorType::kMean:
      total_error =
          std::accumulate(error, error + kNumberOfContactPairsInteract,
                          kNumberOfContactPairsInteract) /
          kNumberOfContactPairsInteract;
      break;
    case ContactKeyframeErrorType::kSum:
      total_error =
          std::accumulate(error, error + kNumberOfContactPairsInteract,
                          kNumberOfContactPairsInteract);
      break;
    case ContactKeyframeErrorType::kNorm:
    default:
      total_error = mju_norm(error, kNumberOfContactPairsInteract);
      break;
  }
  return total_error;
}

bool MotionStrategy::SaveStrategy(const std::string& name,
                                  const std::string& path) {
  std::string filename(path);
  filename.append(name);
  filename.append(".json");

  try {
    json json_keyframes = json::array();
    json json_kf;
    for (auto& kf: contact_keyframes_) {
      to_json(json_kf, kf);
      json_keyframes.push_back(json_kf);
    }

    std::ofstream of(filename);
    if (!of.is_open()) {
      std::printf("\nFailed to open file %s", filename.c_str());
      return false;
    }
    of << std::setw(4) << json_keyframes << std::endl;
    of.close();
    std::printf("\nKeyframe sequence saved as %s", filename.c_str());
    return true;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return false;
  }
}

bool MotionStrategy::LoadStrategy(const std::string& name,
                                  const std::string& path) {
  std::string filename(path);
  filename.append(name);
  filename.append(".json");
  try {
    std::ifstream file(filename);
    if (!file.good()) {
      std::printf("\nFile %s does not exist", filename.c_str());
      return false;
    }
    contact_keyframes_.clear();
    json json_keyframes = json::parse(file);
    ContactKeyframe kf;
    for (auto& json_kf: json_keyframes) {
      from_json(json_kf, kf);
      contact_keyframes_.push_back(kf);
    }

    current_keyframe_index_ = 0;
    std::printf("\nKeyframe sequence loaded from %s", filename.c_str());
    return true;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return false;
  }
}

void to_json(json& j, const MotionStrategy& strategy) {
  j = json{{"contact_keyframes", strategy.GetContactKeyframes()}};
}

void from_json(const json& j, MotionStrategy& strategy) {
  strategy.SetContactKeyframes(
      j.at("contact_keyframes").get<std::vector<ContactKeyframe>>());
}

}  // namespace mjpc::humanoid
