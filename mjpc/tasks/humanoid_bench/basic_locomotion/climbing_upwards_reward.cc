//
// Created by Moritz Meser on 15.05.24.
//

#include "climbing_upwards_reward.h"

#include <cmath>
#include <limits>

#include "mjpc/utilities.h"
#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mujoco/mujoco.h"

namespace mjpc {
    double climbing_upwards_reward(const mjModel *model, const mjData *data, const double walk_speed) {
        // initialize reward
        double reward = 1.0;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double left_foot_height = SensorByName(model, data, "left_foot_height")[2];
        double right_foot_height = SensorByName(model, data, "right_foot_height")[2];
        double standing = tolerance(head_height - left_foot_height, {1.2, INFINITY}, 0.45) *
                          tolerance(head_height - right_foot_height, {1.2, INFINITY}, 0.45);

        reward *= standing;

        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.5, INFINITY}, 1.9, "linear", 0.0);

        reward *= upright;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        reward *= small_control;

        // ----- move speed ----- //
        double com_velocity = SensorByName(model, data, "center_of_mass_velocity")[0];
        double move = tolerance(com_velocity, {walk_speed, INFINITY}, walk_speed, "linear", 0.0);
        move = (5 * move + 1) / 6;

        reward *= move;

        return reward;
    }
} // namespace mjpc