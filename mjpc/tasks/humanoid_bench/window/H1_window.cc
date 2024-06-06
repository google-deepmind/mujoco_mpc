//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_window.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench window task ---------------- //
// ------------------------------------------------------------------------------ //
    void H1_window::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                         double *residual) const {
        // ----- set parameters ---- //
        double const standHeight = 1.65;

        // ----- standing ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double standing = tolerance(head_height, {standHeight, INFINITY}, standHeight / 4);

        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9, "linear", 0.0);

        double stand_reward = standing * upright;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- window contact reward ----- //
        double window_contact_reward = std::numeric_limits<double>::max();
        for (const auto &site_name: {"wipe_contact_site_a", "wipe_contact_site_b", "wipe_contact_site_c",
                                     "wipe_contact_site_d", "wipe_contact_site_e"}) {
            double site_xpos = SensorByName(model, data, site_name)[0];
            window_contact_reward = std::min(window_contact_reward, tolerance(site_xpos, {0.92, 0.92}, 0.4, "linear"));
        }

        // ----- window contact filter ----- //
        double window_contact_filter = 0;

        int const window_pane_id = mj_name2id(model, mjOBJ_GEOM, "window_pane_collision");
        int const window_wipe_id = mj_name2id(model, mjOBJ_GEOM, "window_wipe_collision");

        for (int i = 0; i < data->ncon; i++) {
            if ((data->contact[i].geom1 == window_pane_id && data->contact[i].geom2 == window_wipe_id) ||
                (data->contact[i].geom1 == window_wipe_id && data->contact[i].geom2 == window_pane_id)) {
                window_contact_filter = 1;
                break;
            }
        }

        // ----- hand tool proximity reward ----- //
        double *left_hand_pos = SensorByName(model, data, "left_hand_position");
        double *right_hand_pos = SensorByName(model, data, "right_hand_position");
        double *window_wiping_tool_pos = SensorByName(model, data, "window_wiping_tool");

        double left_hand_tool_distance = std::sqrt(std::pow(left_hand_pos[0] - window_wiping_tool_pos[0], 2) +
                                                   std::pow(left_hand_pos[1] - window_wiping_tool_pos[1], 2) +
                                                   std::pow(left_hand_pos[2] - window_wiping_tool_pos[2], 2));

        double right_hand_tool_distance = std::sqrt(std::pow(right_hand_pos[0] - window_wiping_tool_pos[0], 2) +
                                                    std::pow(right_hand_pos[1] - window_wiping_tool_pos[1], 2) +
                                                    std::pow(right_hand_pos[2] - window_wiping_tool_pos[2], 2));

        double hand_tool_proximity_reward = std::min(tolerance(left_hand_tool_distance, {0, 0.2}, 0.5),
                                                     tolerance(right_hand_tool_distance, {0, 0.2}, 0.5));

        // ----- moving wipe reward ----- //
        double moving_wipe_reward = tolerance(
                std::abs(SensorByName(model, data, "window_wiping_tool_subtreelinvel")[2]), {0.5, 0.5}, 0.5);


        // ----- head window distance reward ----- //
        static double *head_pos0 = nullptr;  // Declare head_pos0 as static

        if (head_pos0 == nullptr) {  // If it's the first call, initialize head_pos0
            head_pos0 = SensorByName(model, data, "head_height");
        }

        double *head_pos = SensorByName(model, data, "head_height");
        double head_window_distance = std::sqrt(std::pow(head_pos[0] - head_pos0[0], 2) +
                                                std::pow(head_pos[1] - head_pos0[1], 2) +
                                                std::pow(head_pos[2] - head_pos0[2], 2));

        double head_window_distance_reward = tolerance(head_window_distance, {0.4, 0.4}, 0.1);


        double manipulation_reward =
                0.2 * (stand_reward * small_control * head_window_distance_reward) + 0.4 * moving_wipe_reward +
                0.4 * hand_tool_proximity_reward;
        double window_contact_total_reward = window_contact_filter * window_contact_reward;

        // ---- reward ---- //
        double reward = 0.5 * manipulation_reward + 0.5 * window_contact_total_reward;


        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench window task -------- //
// ------------------------------------------------------------ //
    void H1_window::TransitionLocked(mjModel *model, mjData *data) {
        //
    }

}  // namespace mjpc
