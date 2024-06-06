//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_push.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench push task ---------------- //
// ---------------------------------------------------------------------------- //
    void H1_push::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
        // ----- define goal position ----- //
        double const goal_pos[] = {1.0, 0.0, 1.0};

        double const hand_dist_penalty = 0.1;
        double const target_dist_penalty = 1.0;
        double const success = 1000;

        // ----- object position ----- //
        double *object_pos = SensorByName(model, data, "object_pos");

        double goal_dist = std::sqrt(std::pow(goal_pos[0] - object_pos[0], 2) +
                                     std::pow(goal_pos[1] - object_pos[1], 2) +
                                     std::pow(goal_pos[2] - object_pos[2], 2));

        double penalty_dist = target_dist_penalty * goal_dist;
        double reward_success = (goal_dist < 0.05) ? success : 0;

        // ----- hand position ----- //
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double hand_dist = std::sqrt(std::pow(left_hand_pos[0] - object_pos[0], 2) +
                                     std::pow(left_hand_pos[1] - object_pos[1], 2) +
                                     std::pow(left_hand_pos[2] - object_pos[2], 2));
        double penalty_hand = hand_dist_penalty * hand_dist;

        // ----- reward ----- //
        double reward = -penalty_hand - penalty_dist + reward_success;

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench push task -------- //
// ------------------------------------------------------------ //
    void H1_push::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc