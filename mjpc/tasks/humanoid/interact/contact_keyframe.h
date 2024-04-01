#ifndef CONTACT_KEYFRAME_H
#define CONTACT_KEYFRAME_H

#include <mujoco/mujoco.h>
#include <vector>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

namespace mjpc::humanoid {

// ---------- Constants ----------------- //
constexpr int NOT_SELECTED = -1;
constexpr int NUMBER_OF_CONTACT_PAIRS = 5;

class ContactPair {
public:
    int body1, body2, geom1, geom2;
    mjtNum local_pos1[3], local_pos2[3];

    ContactPair() : body1(NOT_SELECTED),
                body2(NOT_SELECTED),
                geom1(NOT_SELECTED),
                geom2(NOT_SELECTED),
                local_pos1{0.},
                local_pos2{0.} {}
    
    void Reset();
};

class ContactKeyframe {
public:
    std::string name;
    ContactPair contact_pairs[NUMBER_OF_CONTACT_PAIRS]; 

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