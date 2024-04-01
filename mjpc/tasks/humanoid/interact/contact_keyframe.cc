#include "contact_keyframe.h"

namespace mjpc::humanoid {
    void ContactPair::Reset() {
        body1 = NOT_SELECTED;
        body2 = NOT_SELECTED;
        geom1 = NOT_SELECTED;
        geom2 = NOT_SELECTED;
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