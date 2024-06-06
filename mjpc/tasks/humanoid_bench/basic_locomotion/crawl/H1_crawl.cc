//
// Created by Moritz Meser on 15.05.24.
//

#include "H1_crawl.h"

#include <string>
#include <limits>
#include <cmath>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench crawl task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_crawl::ResidualFn::Residual(const mjModel *model, const mjData *data, double *residual) const {
        double const crawl_height = 0.8;
        double const crawl_speed = 1.0;

        // initialize reward
        double reward = 0.0;

        // ----- small control ----- //
        double small_control = 0.0;
        for (int i = 0; i < model->nu; i++) {
            small_control += tolerance(data->ctrl[i], {0.0, 0.0}, 10.0, "quadratic", 0.0);
        }
        small_control /= model->nu;  // average over all controls
        small_control = (4 + small_control) / 5;

        // ----- move speed ----- //
        double com_velocity = SensorByName(model, data, "center_of_mass_velocity")[0];
        double move = tolerance(com_velocity, {crawl_speed, INFINITY}, 1.0, "linear", 0.0);
        move = (5 * move + 1) / 6;

        // ----- crawling head ----- //
        double head_height = SensorByName(model, data, "head_height")[2];
        double crawling_head = tolerance(head_height, {crawl_height - 0.2, crawl_height + 0.2}, 1.0);

        // ----- crawling ----- //
        double imu_z = SensorByName(model, data, "imu")[2];
        double crawling = tolerance(imu_z, {crawl_height - 0.2, crawl_height + 0.2}, 1.0);

        // ----- reward xquat ----- //
        double *pelvis_quat = SensorByName(model, data, "pelvis_orientation");
        std::array<double, 4> pelvis_quat_array = {pelvis_quat[0], pelvis_quat[1], pelvis_quat[2], pelvis_quat[3]};
        std::array<double, 4> diff_quat{};
        std::transform(pelvis_quat_array.begin(), pelvis_quat_array.end(),
                       std::array<double, 4>{0.75, 0, 0.65, 0}.begin(), diff_quat.begin(), std::minus<double>());
        double norm = std::sqrt(std::inner_product(diff_quat.begin(), diff_quat.end(), diff_quat.begin(), 0.0));
        double reward_xquat = tolerance(norm, {0, 0.0}, 1.0);

        // ----- in tunnel ----- //
        double imu_y = SensorByName(model, data, "imu")[1];
        double in_tunnel = tolerance(imu_y, {-1, 1}, 0);

        reward = (
                         0.1 * small_control
                         + 0.25 * std::min(crawling, crawling_head)
                         + 0.4 * move
                         + 0.25 * reward_xquat
                 ) * in_tunnel;

        residual[0] = std::exp(-reward);
    }
}  // namespace mjpc
