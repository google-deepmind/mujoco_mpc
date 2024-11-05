//
// Created by Moritz Meser on 22.05.24.
//

#include "utility_functions.h"
bool CheckAnyCollision(const mjModel *model, const mjData *data, int body_id) {
  for (int i = 0; i < data->ncon; i++) {
    if (data->contact[i].geom1 == body_id ||
        data->contact[i].geom2 == body_id) {
      return true;
    }
  }
  return false;
}