//
// Created by Moritz Meser on 20.05.24.
//

#include "H1_poles.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/tasks/humanoid_bench/utility/utility_functions.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench poles task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_poles::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                        double *residual) const {
        double const standHeight = 1.65;
        double const moveSpeed = 0.5;

        double standing = tolerance(SensorByName(model, data, "head_height")[2], {standHeight, INFINITY},
                                    standHeight / 4, "linear", 0.0);

        double upright = tolerance(SensorByName(model, data, "torso_upright")[2], {0.9, INFINITY}, 1.9, "linear", 0.0);

        double stand_reward = standing * upright;

        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        double com_velocity = SensorByName(model, data, "center_of_mass_velocity")[0];
        double move = tolerance(com_velocity, {moveSpeed, INFINITY}, moveSpeed, "linear", 0.0);
        move = (5 * move + 1) / 6;

        // ---- wall collision discount ---- //
        double collision_discount = 1.0;
        // Get the ID of the "pole_rows" body
        int pole_rows_id = mj_name2id(model, mjOBJ_BODY, "pole_rows");

        // Iterate over all the child bodies of the "pole_rows" body
        for (int i = 0; i < model->nbody; i++) {
            if (model->body_parentid[i] == pole_rows_id) {
                // Get the ID of the child body
                int child_body_id = i;

                // Iterate over all the geometries
                for (int j = 0; j < model->ngeom; j++) {
                    if (model->geom_bodyid[j] == child_body_id) {
                        // Get the ID of the geometry
                        int geom_id = j;

                        // Check for collisions
                        if (CheckAnyCollision(model, data, geom_id)) {
                            collision_discount = 0.1;
                            break;
                        }
                    }
                }
            }
        }

        // ---- reward computation ---- //
        double reward = (0.5 * (small_control * stand_reward) + 0.5 * move) * collision_discount;

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench poles task -------- //
// ------------------------------------------------------------ //
    void H1_poles::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
