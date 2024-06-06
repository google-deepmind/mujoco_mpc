//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MUJOCO_MPC_CLIMBING_UPWARDS_REWARD_H
#define MUJOCO_MPC_CLIMBING_UPWARDS_REWARD_H

#include "mujoco/mujoco.h"

namespace mjpc {
    double climbing_upwards_reward(const mjModel *model, const mjData *data, double walk_speed);
} // namespace mjpc
#endif //MUJOCO_MPC_CLIMBING_UPWARDS_REWARD_H
