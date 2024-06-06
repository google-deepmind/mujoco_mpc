//
// Created by Moritz Meser on 20.05.24.
//

#include "H1_sit.h"

#include <cmath>
#include <limits>

#include "mjpc/utilities.h"
#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mujoco/mujoco.h"


namespace mjpc {
// ----------------- Residuals for humanoid_bench Sit Simple task ---------------- //
// ---------------------------------------------------------------------------------- //
    void H1_sit::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        // ----- sitting ----- //
        double sitting = tolerance(data->qpos[2], {0.68, 0.72}, 0.2);

        // ----- on chair ----- //
        double chair_location_x = SensorByName(model, data, "chair")[0];
        double chair_location_y = SensorByName(model, data, "chair")[1];
        double on_chair_x = tolerance(data->qpos[0] - chair_location_x, {-0.19, 0.19}, 0.2);
        double on_chair_y = tolerance(data->qpos[1] - chair_location_y, {0.0, 0.0}, 0.1);
        double on_chair = on_chair_x * on_chair_y;

        // ----- sitting posture ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double imu_z = SensorByName(model, data, "imu")[2];
        double sitting_posture = tolerance(head_height - imu_z, {0.35, 0.45}, 0.3);

        // ----- upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.95, INFINITY}, 0.9, "linear", 0.0);

        // ----- sit reward ----- //
        double sit_reward = (0.5 * sitting + 0.5 * on_chair) * upright * sitting_posture;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- don't move ----- //
        double horizontal_velocity_x = SensorByName(model, data, "center_of_mass_velocity")[0];
        double horizontal_velocity_y = SensorByName(model, data, "center_of_mass_velocity")[1];
        double dont_move = (tolerance(horizontal_velocity_x, {0.0, 0.0}, 2) +
                            tolerance(horizontal_velocity_y, {0.0, 0.0}, 2)) / 2;

        // ----- reward ----- //
        double reward = small_control * sit_reward * dont_move;

        // ----- residual ----- //
        residual[0] = std::exp(-reward);
    }
}  // namespace mjpc
