#include "H1_balance.h"

#include <string>
# include <limits>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench balance task ---------------- //
// ------------------------------------------------------------------------------- //
    void Balance_Simple::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                              double *residual) const {
        // ----- set parameters ----- //
        double const standHeight = 1.65;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {standHeight + 0.35, INFINITY}, standHeight / 4);


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        // ----- stand_reward ----- //
        double stand_reward = standing * upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 1.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- horizontal velocity ----- //
        double horizontal_velocity_x = SensorByName(model, data, "center_of_mass_velocity")[0];
        double horizontal_velocity_y = SensorByName(model, data, "center_of_mass_velocity")[1];
        double dont_move = (tolerance(horizontal_velocity_x, {0.0, 0.0}, 2.0) +
                            tolerance(horizontal_velocity_y, {0.0, 0.0}, 2.0)) / 2;

        // ----- reward ----- //
        double reward = stand_reward * small_control * dont_move;

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench balance task -------- //
// --------------------------------------------------------------- //
    void Balance_Simple::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
