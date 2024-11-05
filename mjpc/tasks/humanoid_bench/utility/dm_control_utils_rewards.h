//
// Created by Moritz Meser on 26.04.24.
//

#ifndef MUJOCO_MPC_DM_CONTROL_UTILS_REWARDS_H
#define MUJOCO_MPC_DM_CONTROL_UTILS_REWARDS_H

#include <string>
#include <utility>

//
// this file contains the reimplementation of some of the utility functions from
// dm_control the original implementation is from Google DeepMind and can be
// found here:
// https://github.com/google-deepmind/dm_control/tree/main/dm_control/utils/rewards.py
//

double sigmoid(double x, double value_at_1, std::string sigmoid_type);

double tolerance(double x, std::pair<double, double> bounds = {0.0, 0.0},
                 double margin = 0.0, std::string sigmoid_type = "gaussian",
                 double value_at_margin = 0.1);

#endif  // MUJOCO_MPC_DM_CONTROL_UTILS_REWARDS_H
