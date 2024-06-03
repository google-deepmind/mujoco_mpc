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

#ifndef CONTACT_KEYFRAME_H
#define CONTACT_KEYFRAME_H

#include <mujoco/mujoco.h>
#include <vector>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace mjpc::humanoid {

// ---------- Constants ----------------- //
constexpr int kNotSelectedInteract = -1;
constexpr int kNumberOfContactPairsInteract = 5;

class ContactPair {
public:
    int body1, body2, geom1, geom2;
    mjtNum local_pos1[3], local_pos2[3];

    ContactPair() : body1(kNotSelectedInteract),
                body2(kNotSelectedInteract),
                geom1(kNotSelectedInteract),
                geom2(kNotSelectedInteract),
                local_pos1{0.},
                local_pos2{0.} {}
    
    void Reset();
};

class ContactKeyframe {
public:
    std::string name;
    ContactPair contact_pairs[kNumberOfContactPairsInteract]; 

    // the direction on the xy-plane for the torso to point towards   
    std::vector<mjtNum> facing_target; 
    
    // weight of all residual terms (name -> value map)
    std::map<std::string, mjtNum> weight;

    ContactKeyframe() : name(""),
              contact_pairs{},
              facing_target(),
              weight() {}

    void Reset();
};

void to_json(json& j, const ContactPair& contact_pair);
void from_json(const json& j, ContactPair& contact_pair);
void to_json(json& j, const ContactKeyframe& keyframe);
void from_json(const json& j, ContactKeyframe& keyframe);

}  // namespace mjpc::humanoid

#endif  // CONTACT_KEYFRAME_H
