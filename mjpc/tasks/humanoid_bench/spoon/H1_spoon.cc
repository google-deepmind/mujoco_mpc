//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_spoon.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench spoon task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_spoon::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                        double *residual) const {
        // ----- set parameters ----- //
        double const stand_height = 1.65;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);

        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        double stand_reward = standing * upright;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- hand proximity
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double *right_hand_pos = SensorByName(model, data, "right_hand_position");
        double *spoon_pos = SensorByName(model, data, "spoon_handle");
        double left_hand_tool_distance = std::sqrt(std::pow(left_hand_pos[0] - spoon_pos[0], 2) +
                                                   std::pow(left_hand_pos[1] - spoon_pos[1], 2) +
                                                   std::pow(left_hand_pos[2] - spoon_pos[2], 2));
        double right_hand_tool_distance = std::sqrt(std::pow(right_hand_pos[0] - spoon_pos[0], 2) +
                                                    std::pow(right_hand_pos[1] - spoon_pos[1], 2) +
                                                    std::pow(right_hand_pos[2] - spoon_pos[2], 2));
        double hand_tool_proximity_reward = tolerance(std::min(left_hand_tool_distance, right_hand_tool_distance),
                                                      {0.0, 0.2}, 0.5);

        // ----- spoon ----- //
        double current_spin_angle = data->time * (1.0 / 50) * (2 * M_PI / 40);

        double *spoon_target_pos = new double[3];
        std::array<double, 3> initial_pos = {0.75, -0.1, 0.95};
        std::array<double, 3> spin_offset = {std::cos(current_spin_angle) * 0.06, std::sin(current_spin_angle) * 0.06,
                                             0};

        for (int i = 0; i < 3; i++) {
            spoon_target_pos[i] = initial_pos[i] + spin_offset[i];
        }

        double *spoon_plate_pos = SensorByName(model, data, "spoon_plate");
        double *cup_pos = SensorByName(model, data, "cup");
        double target_distance = std::sqrt(std::pow(spoon_plate_pos[0] - spoon_target_pos[0], 2) +
                                           std::pow(spoon_plate_pos[1] - spoon_target_pos[0], 2) +
                                           std::pow(spoon_plate_pos[2] - spoon_target_pos[2], 2));
        double spoon_spinning_reward = tolerance(target_distance, {0.0, 0.0}, 0.15);

        bool spoon_in_cup_x = std::abs(spoon_plate_pos[0] - cup_pos[0]) < 0.1;
        bool spoon_in_cup_y = std::abs(spoon_plate_pos[1] - cup_pos[1]) < 0.1;
        bool spoon_in_cup_z = std::abs(spoon_plate_pos[2] - (cup_pos[2] + 0.1)) < 0.1;
        double reward_spoon_in_cup = static_cast<double>(spoon_in_cup_x + spoon_in_cup_y + spoon_in_cup_z) / 3;

        // ----- reward computation ----- //
        double reward = (
                0.15 * (stand_reward * small_control)
                + 0.25 * hand_tool_proximity_reward
                + 0.25 * reward_spoon_in_cup
                + 0.35 * spoon_spinning_reward
        );

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench spoon task -------- //
// ------------------------------------------------------------ //
    void H1_spoon::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
