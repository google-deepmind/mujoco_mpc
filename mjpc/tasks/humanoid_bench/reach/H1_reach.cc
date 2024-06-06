#include "H1_reach.h"

#include <string>
#include <limits>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include <array>
#include <random>

namespace mjpc {
// ----------------- Residuals for humanoid_bench reach task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_reach::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double *goal_pos = SensorByName(model, data, "goal_pos");
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double hand_dist = std::sqrt(std::pow(goal_pos[0] - left_hand_pos[0], 2) +
                                     std::pow(goal_pos[1] - left_hand_pos[1], 2) +
                                     std::pow(goal_pos[2] - left_hand_pos[2], 2));

        double healthy_reward = data->xmat[1 * 9 + 8];
        double motion_penalty = 0.0;
        for (int i = 0; i < model->nu; i++) {
            motion_penalty += data->qvel[i];
        }
        double reward_close = (hand_dist < 1) ? 5 : 0;
        double reward_success = (hand_dist < 0.05) ? 10 : 0;

        // ----- reward ----- //
        double reward = healthy_reward - 0.0001 * motion_penalty + reward_close + reward_success;

        // ----- residual ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench reach task -------- //
// ------------------------------------------------------------- //
    void H1_reach::TransitionLocked(mjModel *model, mjData *data) {
        double *goal_pos = SensorByName(model, data, "goal_pos");
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double hand_dist = std::sqrt(std::pow(goal_pos[0] - left_hand_pos[0], 2) +
                                     std::pow(goal_pos[1] - left_hand_pos[1], 2) +
                                     std::pow(goal_pos[2] - left_hand_pos[2], 2));
        // check if task is done
        if (hand_dist < 0.1) {
            // generate new random target
            std::array<double, 3> target_low = {-2, -2, 0.2};
            std::array<double, 3> target_high = {2, 2, 2.0};
            std::random_device rd;
            std::mt19937 gen(rd());
            std::array<double, 3> new_target = {0, 0, 0};
            for (int i = 0; i < 3; ++i) {
                std::uniform_real_distribution<> dis(target_low[i], target_high[i]);
                new_target[i] = dis(gen);
            }
            // copy new target to mocap_pos
            mju_copy3(data->mocap_pos, new_target.data());
        }
    }

}  // namespace mjpc
