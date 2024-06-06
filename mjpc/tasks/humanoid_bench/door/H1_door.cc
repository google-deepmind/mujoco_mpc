//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_door.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench door task ---------------- //
// ---------------------------------------------------------------------------- //
    void H1_door::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
        // ----- set parameters ----- //
        double const standHeight = 1.65;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);

        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 0.9, "linear", 0.0);

        double stand_reward = standing * upright;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- door openness reward ----- //
        double door_openness_reward = std::min(1.0, (data->qpos[model->nq - 2] / 1) *
                                                    std::abs(data->qpos[model->nq - 2] / 1));

        // ----- door hatch openness reward ----- //
        double door_hatch_openness_reward = tolerance(data->qpos[model->nq - 1], {0.75, 2}, 0.75, "linear");

        // ----- hand hatch proximity reward ----- //
        double *door_hatch_pos = SensorByName(model, data, "door_hatch");
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double *right_hand_pos = SensorByName(model, data, "right_hand_position");

        double left_hand_hatch_closeness = std::sqrt(
                std::pow(door_hatch_pos[0] - left_hand_pos[0], 2) +
                std::pow(door_hatch_pos[1] - left_hand_pos[1], 2) +
                std::pow(door_hatch_pos[2] - left_hand_pos[2], 2));
        double right_hand_hatch_closeness = std::sqrt(
                std::pow(door_hatch_pos[0] - right_hand_pos[0], 2) +
                std::pow(door_hatch_pos[1] - right_hand_pos[1], 2) +
                std::pow(door_hatch_pos[2] - right_hand_pos[2], 2));

        double hand_hatch_proximity_reward = tolerance(
                std::min(right_hand_hatch_closeness, left_hand_hatch_closeness),
                {0, 0.25},
                1,
                "linear");

        // ----- passage reward ----- //
        double passage_reward = tolerance(
                SensorByName(model, data, "imu")[0],
                {1.2, INFINITY},
                1,
                "linear",
                0.0);

        // ----- total reward ----- //
        double reward = 0.1 * stand_reward * small_control
                        + 0.45 * door_openness_reward
                        + 0.05 * door_hatch_openness_reward
                        + 0.05 * hand_hatch_proximity_reward
                        + 0.35 * passage_reward;


        // ----- residuals ----- //

        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench door task -------- //
// ------------------------------------------------------------ //
    void H1_door::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
