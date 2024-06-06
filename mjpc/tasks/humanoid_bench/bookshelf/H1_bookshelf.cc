//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_bookshelf.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench bookshelf task ---------------- //
// --------------------------------------------------------------------------------- //
    void H1_bookshelf::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                            double *residual) const {
        // ----- set parameters ----- //
        double const stand_height = 1.65;
        std::vector<int> bookshelf_objects = {-19, -22, -15, -20, -23, -23};
        std::vector<std::vector<double>> placement_goals = {
                {0.75, -0.25, 1.55},
                {0.8,  0.05,  0.95},
                {0.8,  -0.25, 0.95},
                {0.85, 0.05,  0.35},
                {0.85, -0.25, 0.35},
        };

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

        double *curr_reach_obj_pos = data->xpos + model->nbody + bookshelf_objects[task_->task_index_];
        double *curr_placement_goal = placement_goals[task_->task_index_].data();

        double obj_goal_dist = std::sqrt(std::pow(curr_reach_obj_pos[0] - curr_placement_goal[0], 2) +
                                         std::pow(curr_reach_obj_pos[1] - curr_placement_goal[1], 2) +
                                         std::pow(curr_reach_obj_pos[2] - curr_placement_goal[2], 2));
        double reward_proximity = tolerance(obj_goal_dist, {0.0, 0.15}, 1.0, "linear");

        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double left_hand_distance = std::sqrt(std::pow(left_hand_pos[0] - curr_reach_obj_pos[0], 2) +
                                              std::pow(left_hand_pos[1] - curr_reach_obj_pos[1], 2) +
                                              std::pow(left_hand_pos[2] - curr_reach_obj_pos[2], 2));

        double *right_hand_pos = SensorByName(model, data, "right_hand_position");
        double right_hand_distance = std::sqrt(std::pow(right_hand_pos[0] - curr_reach_obj_pos[0], 2) +
                                               std::pow(right_hand_pos[1] - curr_reach_obj_pos[1], 2) +
                                               std::pow(right_hand_pos[2] - curr_reach_obj_pos[2], 2));
        double reward_hand_proximity = std::exp(-std::min(left_hand_distance, right_hand_distance));

        double reward = (
                0.2 * (stand_reward * small_control)
                + 0.4 * reward_proximity
                + 0.4 * reward_hand_proximity
        );

        if (obj_goal_dist < 0.15) {
            reward += 100.0;
        }

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench bookshelf task -------- //
// ------------------------------------------------------------ //
    void H1_bookshelf::TransitionLocked(mjModel *model, mjData *data) {
        std::vector<int> bookshelf_objects = {-19, -22, -15, -20, -23, -23};
        std::vector<std::vector<double>> placement_goals = {
                {0.75, -0.25, 1.55},
                {0.8,  0.05,  0.95},
                {0.8,  -0.25, 0.95},
                {0.85, 0.05,  0.35},
                {0.85, -0.25, 0.35},
        };
        double *curr_reach_obj_pos = data->xpos + model->nbody + bookshelf_objects[task_index_];
        double *curr_placement_goal = placement_goals[task_index_].data();

        double obj_goal_dist = std::sqrt(std::pow(curr_reach_obj_pos[0] - curr_placement_goal[0], 2) +
                                         std::pow(curr_reach_obj_pos[1] - curr_placement_goal[1], 2) +
                                         std::pow(curr_reach_obj_pos[2] - curr_placement_goal[2], 2));
        if (obj_goal_dist < 0.15) {
            task_index_++;
        }
    }

}  // namespace mjpc
