//
// Created by Moritz Meser on 21.05.24.
//

#include "H1_truck.h"
#include <string>
# include <limits>
#include <cmath>
#include <algorithm>

#include "mujoco/mujoco.h"
#include "mjpc/utilities.h"

#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"

namespace mjpc {
// ----------------- Residuals for humanoid_bench truck task ---------------- //
// ----------------------------------------------------------------------------- //
    void H1_truck::ResidualFn::Residual(const mjModel *model, const mjData *data,
                                        double *residual) const {
        // ----- initialize  ----- //
        double reward = 0.0;
        std::map<std::string, int> package_indices_ = {{"package_a", 0},
                                                       {"package_b", 1},
                                                       {"package_c", 2},
                                                       {"package_d", 3},
                                                       {"package_e", 4}};

        // check if packages have been picked up from truck
        for (std::string &s: task_->packages_on_truck_) {
            double height = SensorByName(model, data, s)[2];
            int index = package_indices_[s];
            double initial_height = task_->initial_zs_[index];
            if (height > initial_height + 0.1) {
                reward += 100;
            }
        }

        // check if packages have been placed on table
        for (std::string &s: task_->packages_picked_up_) {
            if (task_->IsPackageUponTable(model, data, s)) {
                reward += 100;
            }
        }

        //Check if packages are no longer on table
        for (std::string &s: task_->packages_upon_table_) {
            if (!task_->IsPackageUponTable(model, data, s)) {
                reward -= 100;
            }
        }

        // ----- torso upright ----- //
        double torso_upright = SensorByName(model, data, "torso_upright")[2];
        double upright = tolerance(torso_upright, {0.9, INFINITY}, 1.9, "linear", 0.0);

        // minimize distance between robot and packages on truck
        double reward_robot_package_truck = 0;
        if (task_->packages_on_truck_.size() > 0) {
            double min_dist = std::numeric_limits<double>::max();
            for (std::string &s: task_->packages_on_truck_) {
                double *package = SensorByName(model, data, s);
                double *free_base = SensorByName(model, data, "pelvis_position");

                double dist_robot_package_truck = std::sqrt(std::pow(free_base[0] - package[0], 2) +
                                                            std::pow(free_base[1] - package[1], 2) +
                                                            std::pow(free_base[2] - package[2], 2));
                if (dist_robot_package_truck < min_dist) {
                    min_dist = dist_robot_package_truck;
                }
            }
            reward_robot_package_truck = tolerance(min_dist, {0.0, 0.2}, 4.0, "linear", 0.0);
        }

        // minimize distance between robot and packages picked up
        double reward_robot_package_picked_up = 0;

        if (task_->packages_picked_up_.size() > 0) {
            double min_dist = std::numeric_limits<double>::max();
            for (std::string &s: task_->packages_picked_up_) {
                double *package = SensorByName(model, data, s);
                double *free_base = SensorByName(model, data, "pelvis_position");

                double dist_robot_package_picked_up = std::sqrt(std::pow(free_base[0] - package[0], 2) +
                                                                std::pow(free_base[1] - package[1], 2) +
                                                                std::pow(free_base[2] - package[2], 2));
                if (dist_robot_package_picked_up < min_dist) {
                    min_dist = dist_robot_package_picked_up;
                }
            }
            reward_robot_package_picked_up = tolerance(min_dist, {0.0, 0.2}, 4.0, "linear", 0.0);
        }

        // minimize distance between picked up packages and table
        double reward_package_table = 0;
        if (task_->packages_picked_up_.size() > 0) {
            double min_dist = std::numeric_limits<double>::max();
            for (std::string &s: task_->packages_picked_up_) {
                double *package = SensorByName(model, data, s);
                double *table = SensorByName(model, data, "table");
                double dist_package_table = std::sqrt(std::pow(table[0] - package[0], 2) +
                                                      std::pow(table[1] - package[1], 2) +
                                                      std::pow(table[2] - package[2], 2));
                if (dist_package_table < min_dist) {
                    min_dist = dist_package_table;
                }
            }
            reward_package_table = tolerance(min_dist, {0.0, 0.2}, 4.0, "linear", 0.0);
        }

        // ----- reward ----- //
        reward += upright * (
                1
                + reward_robot_package_truck
                + reward_robot_package_picked_up
                + reward_package_table
        );

        // ----- residuals ----- //
        residual[0] = std::exp(-reward);
    }

// -------- Transition for humanoid_bench truck task -------- //
// ------------------------------------------------------------ //
    void H1_truck::TransitionLocked(mjModel *model, mjData *data) {
        if (initial_zs_.empty()) {
            // Initialize initial_zs_ here, only done once
            initial_zs_.resize(5);
            initial_zs_[0] = SensorByName(model, data, "package_a")[2];
            initial_zs_[1] = SensorByName(model, data, "package_b")[2];
            initial_zs_[2] = SensorByName(model, data, "package_c")[2];
            initial_zs_[3] = SensorByName(model, data, "package_d")[2];
            initial_zs_[4] = SensorByName(model, data, "package_e")[2];
        }
        // update package states

        std::map<std::string, int> package_indices_ = {{"package_a", 0},
                                                       {"package_b", 1},
                                                       {"package_c", 2},
                                                       {"package_d", 3},
                                                       {"package_e", 4}};

        // check if packages have been picked up from truck
        for (std::string &s: packages_on_truck_) {
            double height = SensorByName(model, data, s)[2];
            int index = package_indices_[s];
            double initial_height = initial_zs_[index];
            if (height > initial_height + 0.1) {
                packages_picked_up_.push_back(s);
                auto tmp = std::remove(packages_on_truck_.begin(), packages_on_truck_.end(), s);
                packages_on_truck_.erase(tmp, packages_on_truck_.end());
            }
        }

        // check if packages have been placed on table
        for (std::string &s: packages_picked_up_) {
            if (IsPackageUponTable(model, data, s)) {
                packages_upon_table_.push_back(s);
                auto tmp = std::remove(packages_picked_up_.begin(), packages_picked_up_.end(), s);
                packages_picked_up_.erase(tmp, packages_picked_up_.end());
            }
        }

        // check if packages are no longer on table
        for (std::string &s: packages_upon_table_) {
            if (!IsPackageUponTable(model, data, s)) {
                packages_picked_up_.push_back(s);
                auto tmp = std::remove(packages_upon_table_.begin(), packages_upon_table_.end(), s);
                packages_upon_table_.erase(tmp, packages_upon_table_.end());
            }
        }
    }

    bool H1_truck::IsPackageUponTable(const mjModel *model, const mjData *data, const std::string &package) const {
        double x = SensorByName(model, data, package)[0];
        double y = SensorByName(model, data, package)[1];
        double z = SensorByName(model, data, package)[2];

        return (x < 2.35 && y < -1.35 && x > 1.65 && y > -2.05 && z > 0.5);
    }
}  // namespace mjpc
