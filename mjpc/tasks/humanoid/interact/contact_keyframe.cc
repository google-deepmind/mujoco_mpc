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

#include "contact_keyframe.h"

namespace mjpc::humanoid {
    void ContactPair::Reset() {
        body1 = kNotSelectedInteract;
        body2 = kNotSelectedInteract;
        geom1 = kNotSelectedInteract;
        geom2 = kNotSelectedInteract;
        for (int i = 0; i < 3; i++) {
            local_pos1[i] = 0.;
            local_pos2[i] = 0.;
        }
    }

    void ContactKeyframe::Reset() {
        name.clear();
        for (auto& contact_pair : contact_pairs)
            contact_pair.Reset();
        
        facing_target.clear();
        weight.clear();
    }

    void to_json(json& j, const ContactPair& contact_pair) {
        j = json{{"body1", contact_pair.body1},
                 {"body2", contact_pair.body2},
                 {"geom1", contact_pair.geom1},
                 {"geom2", contact_pair.geom2},
                 {"local_pos1", contact_pair.local_pos1},
                 {"local_pos2", contact_pair.local_pos2}};
    }

    void from_json(const json& j, ContactPair& contact_pair) {
        j.at("body1").get_to(contact_pair.body1);
        j.at("body2").get_to(contact_pair.body2);
        j.at("geom1").get_to(contact_pair.geom1);
        j.at("geom2").get_to(contact_pair.geom2);
        j.at("local_pos1").get_to(contact_pair.local_pos1);
        j.at("local_pos2").get_to(contact_pair.local_pos2);
    }

    void to_json(json& j, const ContactKeyframe& keyframe) {
        j = json{{"name", keyframe.name},
                 {"contacts", keyframe.contact_pairs},
                 {"facing_target", keyframe.facing_target},
                 {"weight", keyframe.weight}};
    }

    void from_json(const json& j, ContactKeyframe& keyframe) {
        j.at("name").get_to(keyframe.name);
        j.at("contacts").get_to(keyframe.contact_pairs);
        j.at("facing_target").get_to(keyframe.facing_target);
        j.at("weight").get_to(keyframe.weight);
    }
}
