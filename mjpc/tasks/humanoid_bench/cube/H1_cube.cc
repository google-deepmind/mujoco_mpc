//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_cube.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench cube task ---------------- //
// ---------------------------------------------------------------------------- //
    void H1_cube::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
        // ----- set parameters ----- //
        double const standHeight = 1.65;

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

        // ----- don't move reward ----- //
        double *com_vel = SensorByName(model, data, "center_of_mass_velocity");
        double dont_move = (tolerance(com_vel[0], {0.0, 0.0}, 2.0, "quadratic", 0.0) +
                            tolerance(com_vel[1], {0.0, 0.0}, 2.0, "quadratic", 0.0)) / 2;

        // ----- orientation alignment reward ----- //
        double *left_cube_orientation = SensorByName(model, data, "left_cube_quat");
        double *right_cube_orientation = SensorByName(model, data, "right_cube_quat");
        double *target_cube_orientation = SensorByName(model, data, "target_cube_quat");

        double left_orientation_alignment_reward = tolerance(
                std::sqrt(std::pow(left_cube_orientation[0] - target_cube_orientation[0], 2) +
                          std::pow(left_cube_orientation[1] - target_cube_orientation[1], 2) +
                          std::pow(left_cube_orientation[2] - target_cube_orientation[2], 2) +
                          std::pow(left_cube_orientation[3] - target_cube_orientation[3], 2)), {0.0, 0.0}, 0.3);

        double right_orientation_alignment_reward = tolerance(
                std::sqrt(std::pow(right_cube_orientation[0] - target_cube_orientation[0], 2) +
                          std::pow(right_cube_orientation[1] - target_cube_orientation[1], 2) +
                          std::pow(right_cube_orientation[2] - target_cube_orientation[2], 2) +
                          std::pow(right_cube_orientation[3] - target_cube_orientation[3], 2)), {0.0, 0.0}, 0.3);

        double orientation_alignment_reward =
                (left_orientation_alignment_reward + right_orientation_alignment_reward) / 2;

        // ----- cube closeness reward ----- //
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double *right_hand_pos = SensorByName(model, data, "right_hand_position");
        double *left_cube_pos = SensorByName(model, data, "left_cube_position");
        double *right_cube_pos = SensorByName(model, data, "right_cube_position");

        double left_hand_cube_distance = std::sqrt(
                std::pow(left_hand_pos[0] - left_cube_pos[0], 2) +
                std::pow(left_hand_pos[1] - left_cube_pos[1], 2) +
                std::pow(left_hand_pos[2] - left_cube_pos[2], 2));

        double right_hand_cube_distance = std::sqrt(
                std::pow(right_hand_pos[0] - right_cube_pos[0], 2) +
                std::pow(right_hand_pos[1] - right_cube_pos[1], 2) +
                std::pow(right_hand_pos[2] - right_cube_pos[2], 2));

        double left_hand_cube_proximity = tolerance(left_hand_cube_distance, {0, 0.1}, 0.5);
        double right_hand_cube_proximity = tolerance(right_hand_cube_distance, {0, 0.1}, 0.5);

        double cube_closeness_reward = (left_hand_cube_proximity + right_hand_cube_proximity) / 2;

        // ----- total reward ----- //
        double reward = 0.2 * (small_control * stand_reward * dont_move)
                        + 0.5 * orientation_alignment_reward
                        + 0.3 * cube_closeness_reward;



        // ----- residuals ----- //

        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench cube task -------- //
// ------------------------------------------------------------ //
    void H1_cube::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
