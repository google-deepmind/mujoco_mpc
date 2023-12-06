#include "mjpc/tasks/allegro_cube/allegro_cube.h"
#include <iostream>

#include <cmath>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string AllegroCube::XmlPath() const {
  return GetModelPath("allegro_cube/task.xml");
}
std::string AllegroCube::Name() const { return "AllegroCube"; }

// ------- Residuals for cube manipulation task ------
//     Control:  Control should be small
// ------------------------------------------
void AllegroCube::ResidualFn::Residual(const mjModel* model, const mjData* data,
                        double* residual) const {
  int counter = 0;

  // ---------- Control ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;
}

void AllegroCube::TransitionLocked(mjModel* model, mjData* data) {
  // Check for contact between the cube and the floor
  int cube = mj_name2id(model, mjOBJ_GEOM, "cube");
  int floor = mj_name2id(model, mjOBJ_GEOM, "floor");
  
  bool on_floor = false;
  for (int i=0; i < data->ncon; i++) {
    mjContact* g = data->contact + i;
    if ((g->geom1 == cube && g->geom2 == floor) ||
        (g->geom2 == cube && g->geom1 == floor)) {
      on_floor = true;
      break;
    }
  }

  // If the cube is on the floor and not moving, reset it
  double* cube_lin_vel = SensorByName(model, data, "cube_linear_velocity");
  if (on_floor && mju_norm3(cube_lin_vel) < 0.001) {
    int cube_body = mj_name2id(model, mjOBJ_BODY, "cube");
    if (cube_body != -1) {
      int jnt_qposadr = model->jnt_qposadr[model->body_jntadr[cube_body]];
      int jnt_veladr = model->jnt_dofadr[model->body_jntadr[cube_body]];
      mju_copy(data->qpos + jnt_qposadr, model->qpos0 + jnt_qposadr, 7);
      mju_zero(data->qvel + jnt_veladr, 6);
    }

    // Step the simulation forward
    mutex_.unlock();
    mj_forward(model, data);
    mutex_.lock();
  }

}

}  // namespace mjpc
