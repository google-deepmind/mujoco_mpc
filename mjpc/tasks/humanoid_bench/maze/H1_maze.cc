//
// Created by Moritz Meser on 20.05.24.
//

#include "H1_maze.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/tasks/humanoid_bench/utility/utility_functions.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench maze task ---------------- //
// ---------------------------------------------------------------------------- //
    void H1_maze::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
        // ----- set parameters ----- //
        double const standHeight = 1.65;
        double const moveSpeed = 2.0;


        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        double standReward = standing * upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- wall collision ----- //
        double wall_collision_discount = 1.0;
        int maze_id = mj_name2id(model, mjOBJ_BODY, "maze");

        // Iterate over all the child bodies of the "maze" body
        for (int i = 0; i < model->nbody; i++) {
            if (model->body_parentid[i] == maze_id) {
                // Get the ID of the child body
                int child_body_id = i;

                // Iterate over all the geometries
                for (int j = 0; j < model->ngeom; j++) {
                    if (model->geom_bodyid[j] == child_body_id) {
                        // Get the ID of the geometry
                        int geom_id = j;

                        // Check for collisions
                        if (CheckAnyCollision(model, data, geom_id)) {
                            wall_collision_discount = 0.1;
                            break;
                        }
                    }
                }
            }
        }


        // ----- stage convert reward ----- //
        double stage_convert_reward = 0.0;
        double *goal_pos = data->mocap_pos;
        double *pelvis_pos = SensorByName(model, data, "pelvis_position");
        double dist = std::sqrt(std::pow(goal_pos[0] - pelvis_pos[0], 2) +
                                std::pow(goal_pos[1] - pelvis_pos[1], 2) +
                                std::pow(goal_pos[2] - pelvis_pos[2], 2));

        // check if task is done
        if (dist < 0.1) {
            stage_convert_reward = 100.0;
        }

        // ----- move speed ----- //
        double move_direction[3] = {0, 0, 0};

        // Check each case
        if (task_->curr_goal_idx_ == 0) {
            // checkpoint is 0,0,1
            move_direction[0] = 1;
            move_direction[1] = 0;
            move_direction[2] = 0;
        } else if (task_->curr_goal_idx_ == 1) {
           // checkpoint is 3,0,1
            move_direction[0] = 1;
            move_direction[1] = 0;
            move_direction[2] = 0;
        } else if (task_->curr_goal_idx_ == 2) {
            // checkpoint is 3,6,1
            move_direction[0] = 0;
            move_direction[1] = 1;
            move_direction[2] = 0;
        } else if (task_->curr_goal_idx_ == 3) {
            // checkpoint is 6,6,1
            move_direction[0] = 1;
            move_direction[1] = 0;
            move_direction[2] = 0;
        } else {
            // last checkpoint is reached
            move_direction[0] = 0;
            move_direction[1] = 0;
            move_direction[2] = 0;
        }

        // Get the center of mass velocity
        double *com_velocity = SensorByName(model, data, "center_of_mass_velocity");

        double move;
        if (task_->curr_goal_idx_ == 4) {
            // last checkpoint is reached
            move = 1.0;
        } else {
            // Calculate the move reward
            move = tolerance(com_velocity[0] - move_direction[0] * moveSpeed, {0, 0}, 1.0, "linear", 0.0) *
                   tolerance(com_velocity[1] - move_direction[1] * moveSpeed, {0, 0}, 1.0, "linear", 0.0);
        }
        move = (5 * move + 1) / 6;
        // ----- checkpoint proximity ----- //
        double checkpoint_proximity = std::sqrt(std::pow(goal_pos[0] - pelvis_pos[0], 2) +
                                                std::pow(goal_pos[1] - pelvis_pos[1], 2));
        double checkpoint_proximity_reward = tolerance(checkpoint_proximity, {0, 0.0}, 1.0);

        // ----- reward ----- //
        double reward = (0.2 * (standReward * small_control)
                         + 0.4 * move
                         + 0.4 * checkpoint_proximity_reward
                        ) * wall_collision_discount + stage_convert_reward;

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench maze task -------- //
// ------------------------------------------------------------ //
    void H1_maze::TransitionLocked(mjModel *model, mjData *data) {
        double *goal_pos = model->key_mpos + 3 * (curr_goal_idx_ + 1);  // offset 1 is on purpose
        double *pelvis_pos = SensorByName(model, data, "pelvis_position");
        double dist = std::sqrt(std::pow(goal_pos[0] - pelvis_pos[0], 2) +
                                std::pow(goal_pos[1] - pelvis_pos[1], 2));


        // check if task is done
        if (dist < 0.1) {
            curr_goal_idx_ = std::min(curr_goal_idx_ + 1, 4);
        }
        mju_copy3(data->mocap_pos, model->key_mpos + 3 * (curr_goal_idx_ + 1)); // offset 1 is on purpose
    }

    void H1_maze::ResetLocked(const mjModel *model) {
        curr_goal_idx_ = 0;
    }

}  // namespace mjpc
