//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_powerlift.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench powerlift task ---------------- //
// --------------------------------------------------------------------------------- //
    void H1_powerlift::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                            double *residual) const {
        double const standHeight = 1.65;

        double standing = tolerance(SensorByName(model, data, "head_height")[2], {standHeight, INFINITY},
                                    standHeight / 3);

        double upright = tolerance(SensorByName(model, data, "torso_upright")[2], {0.8, INFINITY}, 1.9, "linear", 0.0);

        double stand_reward = standing * upright;

        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        double dumbbell_height = SensorByName(model, data, "dumbbell_height")[2];
        double reward_dumbbell_lifted = tolerance(dumbbell_height, {1.9, 2.1}, 2.0);

        double reward = 0.2 * (small_control * stand_reward) + 0.8 * reward_dumbbell_lifted;

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench powerlift task -------- //
// ------------------------------------------------------------ //
    void H1_powerlift::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc