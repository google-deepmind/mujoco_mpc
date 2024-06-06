//
// Created by Moritz Meser on 22.05.24.
//

#ifndef MUJOCO_MPC_UTILITY_FUNCTIONS_H
#define MUJOCO_MPC_UTILITY_FUNCTIONS_H

#include "mujoco/mjmodel.h"
#include "mujoco/mjdata.h"

bool CheckAnyCollision(const mjModel *pModel, const mjData *pData, int id);

#endif //MUJOCO_MPC_UTILITY_FUNCTIONS_H
