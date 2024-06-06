//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_insert.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench insert task ---------------- //
// ------------------------------------------------------------------------------ //
    void H1_insert::ResidualFn::Residual(const mjModel *model, const mjData *data,
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

        // ----- cube rewards ----- //
        std::vector<double> cube_targets;
        for (const std::string &ch: {"a", "b"}) {
            double *block_peg_pos = SensorByName(model, data, "block_peg_" + ch);
            double *peg_pos = SensorByName(model, data, "peg_" + ch);
            double distance = std::sqrt(std::pow(block_peg_pos[0] - peg_pos[0], 2) +
                                        std::pow(block_peg_pos[1] - peg_pos[1], 2) +
                                        std::pow(block_peg_pos[2] - peg_pos[2], 2));
            cube_targets.push_back(tolerance(distance, {0.0, 0.0}, 0.5, "linear", 0.0));
        }

        double cube_target_reward =
                std::accumulate(cube_targets.begin(), cube_targets.end(), 0.0) / cube_targets.size();

        std::vector<double> peg_heights;
        for (const std::string &ch: {"a", "b"}) {
            double peg_height = SensorByName(model, data, "peg_" + ch)[2];
            peg_heights.push_back(tolerance(peg_height - 1.1, {0.0, 0.0}, 0.15, "linear", 0.0));
        }
        double peg_height_reward = std::accumulate(peg_heights.begin(), peg_heights.end(), 0.0) / peg_heights.size();

        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double *right_hand_pos = SensorByName(model, data, "right_hand_position");
        double *peg_a_pos = SensorByName(model, data, "peg_a");
        double *peg_b_pos = SensorByName(model, data, "peg_b");
        double left_hand_tool_distance = std::sqrt(std::pow(left_hand_pos[0] - peg_a_pos[0], 2) +
                                                   std::pow(left_hand_pos[1] - peg_a_pos[1], 2) +
                                                   std::pow(left_hand_pos[2] - peg_a_pos[2], 2));
        double right_hand_tool_distance = std::sqrt(std::pow(right_hand_pos[0] - peg_b_pos[0], 2) +
                                                    std::pow(right_hand_pos[1] - peg_b_pos[1], 2) +
                                                    std::pow(right_hand_pos[2] - peg_b_pos[2], 2));
        double hand_tool_proximity_reward = tolerance(std::min(left_hand_tool_distance, right_hand_tool_distance),
                                                      {0.0, 0.2}, 0.5);

        // ----- reward computation ----- //
        double reward = (0.5 * (small_control * stand_reward) + 0.5 * cube_target_reward) *
                        (0.5 * peg_height_reward + 0.5 * hand_tool_proximity_reward);

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench insert task -------- //
// ------------------------------------------------------------ //
    void H1_insert::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
