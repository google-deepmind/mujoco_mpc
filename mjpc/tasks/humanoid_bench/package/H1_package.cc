
#include "H1_package.h"

#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench package task ---------------- //
// ------------------------------------------------------------------------------- //
    void H1_package::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                          double *residual) const {
        // ----- set parameters ----- //
        double const stand_height = 1.65;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {stand_height, INFINITY}, stand_height / 4);


        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        double stand_reward = standing * upright;


        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- rewards specific to the package task ----- //

        double *package_location = SensorByName(model, data, "package_location");
        double *package_destination = SensorByName(model, data, "package_destination");
        double *left_hand_location = SensorByName(model, data, "left_hand_position");
        double *right_hand_location = SensorByName(model, data, "right_hand_position");

        double dist_package_destination = std::hypot(
                std::hypot(package_location[0] - package_destination[0], package_location[1] - package_destination[1]),
                package_location[2] - package_destination[2]);

        double dist_hand_package_right = std::hypot(
                std::hypot(right_hand_location[0] - package_location[0], right_hand_location[1] - package_location[1]),
                right_hand_location[2] - package_location[2]);

        double dist_hand_package_left = std::hypot(
                std::hypot(left_hand_location[0] - package_location[0], left_hand_location[1] - package_location[1]),
                left_hand_location[2] - package_location[2]);

        double package_height = std::min(package_location[2], 1.0);

        bool reward_success = dist_package_destination < 0.1;

        // ----- reward computation ----- //
        double reward = (
                stand_reward * small_control
                - 3 * dist_package_destination
                - (dist_hand_package_left + dist_hand_package_right) * 0.1
                + package_height
                + reward_success * 1000
        );

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench package task --------
// for a more complex task this might be necessary (like packageing to different targets)
// ---------------------------------------------
    void H1_package::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
