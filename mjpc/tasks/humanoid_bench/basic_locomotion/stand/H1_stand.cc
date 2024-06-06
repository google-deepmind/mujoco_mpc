//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_stand.h"

#include <string>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/basic_locomotion/walk_reward.h"


namespace mjpc {
// ----------------- Residuals for humanoid_bench stand task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_stand::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const walk_speed = 0.0;
        double const stand_height = 1.65;

        double reward = walk_reward(model, data, walk_speed, stand_height);
        residual[0] = std::exp(-reward);

    }
}  // namespace mjpc
