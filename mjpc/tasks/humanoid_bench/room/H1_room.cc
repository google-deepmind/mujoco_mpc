//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_room.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench room task ---------------- //
// ---------------------------------------------------------------------------- //
    void H1_room::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                       double *residual) const {
        // ----- set parameters ----- //
        double standHeight = 1.65;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);



        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9);

        double standReward = standing * upright;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;


        // ----- room object organization ----- //
        std::vector<std::string> object_names = {
                "chair",
                "trophy",
                "headphone",
                "package_a",
                "package_b",
                "snow_globe"
        };

        std::vector<double> room_object_positions_x;
        std::vector<double> room_object_positions_y;

        for (const auto &object: object_names) {
            double *object_pos = SensorByName(model, data, object);
            room_object_positions_x.push_back(object_pos[0]);
            room_object_positions_y.push_back(object_pos[1]);
        }
        double x_sum = std::accumulate(room_object_positions_x.begin(), room_object_positions_x.end(), 0.0);
        double y_sum = std::accumulate(room_object_positions_y.begin(), room_object_positions_y.end(), 0.0);
        double x_mean = x_sum / room_object_positions_x.size();
        double y_mean = y_sum / room_object_positions_y.size();
        double x_var = 0.0;
        double y_var = 0.0;
        for (int i = 0; i < room_object_positions_x.size(); i++) {
            x_var += std::pow(room_object_positions_x[i] - x_mean, 2);
            y_var += std::pow(room_object_positions_y[i] - y_mean, 2);
        }
        x_var /= room_object_positions_x.size();
        y_var /= room_object_positions_y.size();

        double room_object_organized = tolerance(std::max(x_var, y_var), {0.0, 0.0}, 3.0);

        // ----- reward ----- //
        double reward = 0.2 * (small_control * standReward) + 0.8 * room_object_organized;

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench room task -------- //
// ------------------------------------------------------------ //
    void H1_room::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
