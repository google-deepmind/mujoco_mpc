//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_run.h"

#include <string>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/basic_locomotion/walk_reward.h"


namespace mjpc {
// ----------------- Residuals for humanoid_bench run task ---------------- //
// --------------------------------------------------------------------------- //
    void H1_run::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const walk_speed = 5.0;
        double const stand_height = 1.65;

        double reward = walk_reward(model, data, walk_speed, stand_height);
        residual[0] = std::exp(-reward);
    }
}  // namespace mjpc
