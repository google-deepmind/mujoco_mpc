//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_basketball.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench basketball task ---------------- //
// ---------------------------------------------------------------------------------- //
    void H1_basketball::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                             double *residual) const {
        // ----- set parameters ----- //
        double const standHeight = 1.65;

        // Initialize reward
        double reward = 0.0;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);

        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9, "linear", 0.0);

        double stand_reward = standing * upright;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- hand proximity reward ----- //
        double *basketball_pos = SensorByName(model, data, "basketball");

        // Get the position vectors for left hand and right hand
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double *right_hand_pos = SensorByName(model, data, "right_hand_position");

        // Compute the Euclidean distance from each hand to the basketball
        double left_hand_distance = std::sqrt(
                std::pow(left_hand_pos[0] - basketball_pos[0], 2) +
                std::pow(left_hand_pos[1] - basketball_pos[1], 2) +
                std::pow(left_hand_pos[2] - basketball_pos[2], 2));
        double right_hand_distance = std::sqrt(
                std::pow(right_hand_pos[0] - basketball_pos[0], 2) +
                std::pow(right_hand_pos[1] - basketball_pos[1], 2) +
                std::pow(right_hand_pos[2] - basketball_pos[2], 2));
        double reward_hand_proximity = tolerance(std::max(left_hand_distance, right_hand_distance), {0, 0.2}, 1);

        // ----- ball success reward ----- //
        double ball_hoop_distance = std::sqrt(
                std::pow(basketball_pos[0] - SensorByName(model, data, "hoop_center")[0], 2) +
                std::pow(basketball_pos[1] - SensorByName(model, data, "hoop_center")[1], 2) +
                std::pow(basketball_pos[2] - SensorByName(model, data, "hoop_center")[2], 2));
        double reward_ball_success = tolerance(ball_hoop_distance, {0.0, 0.0}, 7, "linear");

        // ----- stage ----- //
        static std::string stage = "catch";
        if (stage == "catch") {
            int const ball_collision_id = mj_name2id(model, mjOBJ_GEOM, "basketball_collision");
            for (int i = 0; i < data->ncon; i++) {
                if (data->contact[i].geom1 == ball_collision_id || data->contact[i].geom2 == ball_collision_id) {
                    stage = "throw";
                    break;
                }
            }
        }

        if (stage == "throw") {
            reward = 0.15 * (stand_reward * small_control) + 0.05 * reward_hand_proximity +
                     0.8 * reward_ball_success;
        } else if (stage == "catch") {
            reward = 0.5 * (stand_reward * small_control) + 0.5 * reward_hand_proximity;
        }

        if (ball_hoop_distance < 0.05) {
            reward += 1000;
        }

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench basketball task -------- //
// ------------------------------------------------------------ //
    void H1_basketball::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
