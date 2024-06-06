//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_highbar.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench highbar task ---------------- //
// ------------------------------------------------------------------------------- //
    void H1_highbar::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                          double *residual) const {
        // ---- upright reward ---- //
        double upright_reward = tolerance(-SensorByName(model, data, "torso_upright")[2], {0.9, INFINITY}, 1.9,
                                          "linear", 0.0);

        // ---- feet reward ---- //
        double left_foot_height = SensorByName(model, data, "left_foot_height")[2];
        double right_foot_height = SensorByName(model, data, "right_foot_height")[2];
        double feet_reward = tolerance((left_foot_height + right_foot_height) / 2, {4.8, INFINITY}, 2.0, "linear", 0.0);
        feet_reward = (1 + feet_reward) / 2;

        // ---- small control reward ---- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ---- reward ---- //
        double reward = upright_reward * feet_reward * small_control;

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench highbar task -------- //
// ------------------------------------------------------------ //
    void H1_highbar::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc