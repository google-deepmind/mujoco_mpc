//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_cabinet.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench cabinet task ---------------- //
// ------------------------------------------------------------------------------- //
    void H1_cabinet::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                          double *residual) const {
        // ----- set parameters ----- //
        double standHeight = 1.65;


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

        double stabilizationReward = standReward * small_control;

        // ----- subtasks ----- //
        double subtaskReward = 0.0;
        bool subtaskComplete = false;

        if (task_->current_subtask_ == 1) {
            // subtask 1: open cabinet door
            double pulling_cabinet_joint_pos = data->qpos[(4 * 7) + 2];
            double door_openness_reward = std::abs(pulling_cabinet_joint_pos / 0.4);
            subtaskComplete = door_openness_reward > 0.95;
            subtaskReward = door_openness_reward;
        } else if (task_->current_subtask_ == 2) {
            // subtask 2: open drawer
            double drawer_joint_pos = data->qpos[model->nq - (4 * 7) - 5];
            double door_openness_reward = abs(drawer_joint_pos / 0.45);
            subtaskComplete = door_openness_reward > 0.95;
            subtaskReward = door_openness_reward;
        } else if (task_->current_subtask_ == 3) {
            // subtask 3: move cube into cabinet
            double *drawer_cube_pos = SensorByName(model, data, "drawer_cube_pos");

            double normal_cabinet_left_joint_pos = data->qpos[model->nq - (4 * 7) - 4];
            double normal_cabinet_right_joint_pos = data->qpos[model->nq - (4 * 7) - 3];
            double left_door_openness_reward = std::min(1.0, std::abs(normal_cabinet_left_joint_pos));
            double right_door_openness_reward = std::min(1.0, std::abs(normal_cabinet_right_joint_pos));
            double door_openness_reward = std::max(left_door_openness_reward,
                                                   right_door_openness_reward); // any open door is sufficient

            double cube_proximity_horizontal = (tolerance(drawer_cube_pos[0], {-0.9, -0.3}, 0.3, "linear") +
                                                tolerance(drawer_cube_pos[1], {-0.6, 0.6}, 0.3, "linear")) / 2;
            double cube_proximity_vertical = tolerance(drawer_cube_pos[2] - 0.94, {-0.15, 0.15}, 0.3, "linear");

            bool in_cabinet_x = 0.9 - 0.3 <= drawer_cube_pos[0] && drawer_cube_pos[0] <= 0.9 + 0.3;
            bool in_cabinet_y = 0 - 0.6 <= drawer_cube_pos[1] && drawer_cube_pos[1] <= 0 + 0.6;
            bool in_cabinet_z = 0.94 - 0.15 <= drawer_cube_pos[2] && drawer_cube_pos[2] <= 0.94 + 0.15;
            subtaskComplete = in_cabinet_x && in_cabinet_y && in_cabinet_z;

            double drawer_cube_proximity_reward = 0.3 * cube_proximity_horizontal + 0.7 * cube_proximity_vertical;
            stabilizationReward = 0.5 * drawer_cube_proximity_reward + 0.5 * door_openness_reward;
        } else if (task_->current_subtask_ == 4) {
            // subtask 4: close drawer
            double *pullup_drawer_cube_pos = SensorByName(model, data, "pullup_drawer_cube_pos");

            double pullup_drawer_joint_pos = data->qpos[model->nq - (4 * 7) - 1];
            double door_openness_reward = std::min(1.0, std::abs(pullup_drawer_joint_pos));

            //
            // The secondary_door_openness_reward is computed in the reference python implementation, but not used.
            // I still put it here for completeness.
            //
            // double normal_cabinet_left_joint_pos = data->qpos[model->nq - (4 * 7) - 4];
            // double normal_cabinet_right_joint_pos = data->qpos[model->nq - (4 * 7) - 3];
            // double left_door_openness_reward = std::min(1.0, std::abs(normal_cabinet_left_joint_pos));
            //  double right_door_openness_reward = std::min(1.0, std::abs(normal_cabinet_right_joint_pos));
            //  double secondary_door_openness_reward = std::max(left_door_openness_reward, right_door_openness_reward);

            double cube_proximity_horizontal = (tolerance(pullup_drawer_cube_pos[0] - 0.9, {-0.3, 0.3}, 0.3, "linear") +
                                                tolerance(pullup_drawer_cube_pos[1], {-0.6, 0.6}, 0.3, "linear")) / 2;
            double cube_proximity_vertical = tolerance(pullup_drawer_cube_pos[2] - 1.54, {-0.15, 0.15}, 0.3, "linear");

            bool in_cabinet_x = 0.9 - 0.3 <= pullup_drawer_cube_pos[0] && pullup_drawer_cube_pos[0] <= 0.9 + 0.3;
            bool in_cabinet_y = 0 - 0.6 <= pullup_drawer_cube_pos[1] && pullup_drawer_cube_pos[1] <= 0 + 0.6;
            bool in_cabinet_z = 1.54 - 0.15 <= pullup_drawer_cube_pos[2] && pullup_drawer_cube_pos[2] <= 1.54 + 0.15;
            subtaskComplete = in_cabinet_x && in_cabinet_y && in_cabinet_z;

            double drawer_cube_proximity_reward = 0.3 * cube_proximity_horizontal / 2 + 0.7 * cube_proximity_vertical;
            subtaskReward = 0.5 * drawer_cube_proximity_reward + 0.5 * door_openness_reward;
        } else { // all subtasks are complete
            subtaskReward = 1000.0;
            subtaskComplete = false;
        }

        // ----- reward ----- //
        double reward;
        if (task_->current_subtask_ < 5) {
            reward = 0.2 * stabilizationReward + 0.8 * subtaskReward;
        } else {
            reward = subtaskReward;
        }

        if (subtaskComplete) {
            reward += 100.0 * task_->current_subtask_;
        }

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

    // -------- Transition for humanoid_bench cabinet task -------- //
    // ------------------------------------------------------------ //
    void H1_cabinet::TransitionLocked(mjModel *model, mjData *data) {
        bool subtaskComplete = false;

        if (current_subtask_ == 1) {
            // subtask 1: open cabinet door
            double pulling_cabinet_joint_pos = data->qpos[(4 * 7) + 2];
            double door_openness_reward = std::abs(pulling_cabinet_joint_pos / 0.4);
            subtaskComplete = door_openness_reward > 0.95;
        } else if (current_subtask_ == 2) {
            // subtask 2: open drawer
            double drawer_joint_pos = data->qpos[model->nq - (4 * 7) - 5];
            double door_openness_reward = abs(drawer_joint_pos / 0.45);
            subtaskComplete = door_openness_reward > 0.95;
        } else if (current_subtask_ == 3) {
            // subtask 3: move cube into cabinet
            double *drawer_cube_pos = SensorByName(model, data, "drawer_cube_pos");

            bool in_cabinet_x = 0.9 - 0.3 <= drawer_cube_pos[0] && drawer_cube_pos[0] <= 0.9 + 0.3;
            bool in_cabinet_y = 0 - 0.6 <= drawer_cube_pos[1] && drawer_cube_pos[1] <= 0 + 0.6;
            bool in_cabinet_z = 0.94 - 0.15 <= drawer_cube_pos[2] && drawer_cube_pos[2] <= 0.94 + 0.15;

            subtaskComplete = in_cabinet_x && in_cabinet_y && in_cabinet_z;
        } else if (current_subtask_ == 4) {
            // subtask 4: close drawer
            double *pullup_drawer_cube_pos = SensorByName(model, data, "pullup_drawer_cube_pos");

            bool in_cabinet_x = 0.9 - 0.3 <= pullup_drawer_cube_pos[0] && pullup_drawer_cube_pos[0] <= 0.9 + 0.3;
            bool in_cabinet_y = 0 - 0.6 <= pullup_drawer_cube_pos[1] && pullup_drawer_cube_pos[1] <= 0 + 0.6;
            bool in_cabinet_z = 1.54 - 0.15 <= pullup_drawer_cube_pos[2] && pullup_drawer_cube_pos[2] <= 1.54 + 0.15;

            subtaskComplete = in_cabinet_x && in_cabinet_y && in_cabinet_z;
        } else { // all subtasks are complete
            subtaskComplete = false;
        }

        if (subtaskComplete) {
            current_subtask_++;
        }
    }

    void H1_cabinet::ResetLocked(const mjModel *model) {
        current_subtask_ = 1;
    }
}  // namespace mjpc