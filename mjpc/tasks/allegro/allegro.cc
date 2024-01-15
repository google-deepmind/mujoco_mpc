#include "mjpc/tasks/allegro/allegro.h"
#include <iostream>

#include <cmath>
#include <string>
#include <vector>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
std::string AllegroCube::XmlPath() const {
  return GetModelPath("allegro/task.xml");
}
std::string AllegroCube::Name() const { return "AllegroCube"; }

// ------- Residuals for cube manipulation task ------
//     Cube position: (3)
//     Cube orientation: (3)
//     Cube linear velocity: (3)
//     Control:  u
// ------------------------------------------
void AllegroCube::ResidualFn::Residual(const mjModel* model, const mjData* data,
                        double* residual) const {
  int counter = 0;

  // ---------- Cube position ----------
  // TODO(vincekurtz): specify goal position in a better way
  double* cube_position = SensorByName(model, data, "cube_position");
  std::vector<double> goal_cube_position = {0.3, 0.0, 0.025};

  mju_sub3(residual + counter, cube_position, goal_cube_position.data());
  counter += 3;

  // ---------- Cube orientation ----------
  double* cube_orientation = SensorByName(model, data, "cube_orientation");
  double* goal_cube_orientation = SensorByName(model, data, "cube_goal_orientation");
  mju_normalize4(goal_cube_orientation);

  mju_subQuat(residual + counter, goal_cube_orientation, cube_orientation);
  counter += 3;

  // ---------- Cube linear velocity ----------
  double* cube_linear_velocity =
      SensorByName(model, data, "cube_linear_velocity");

  mju_copy(residual + counter, cube_linear_velocity, 3);
  counter += 3;

  // ---------- Control ----------
  mju_copy(residual + counter, data->actuator_force, model->nu);
  counter += model->nu;

  // ---------- Residual () ----------
  mju_sub(residual + counter, data->qpos + 7, model->key_qpos + 7, 16);
  counter += 16;

  // ---------- Residual () ----------
  mju_copy(residual + counter, data->qvel + 6, 16);
  counter += 16;

  // Sanity check
  CheckSensorDim(model, counter);
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

// Change the friction coefficient of all the objects in the scene
void AllegroCube::DomainRandomize(std::vector<mjModel*>& randomized_models) 
    const {
  absl::BitGen gen_;

  // Standard deviation of the friction coefficient change
  const double std_dev = 0.1;

  // Each model has all friction coefficients boosted or shrunk, so some models
  // are more slippery and others are more grippy.
  for (int i=0; i < randomized_models.size(); i++) {
    mjModel* model = randomized_models[i];

    const double friction_change = absl::Gaussian<double>(gen_, 0.0, std_dev);
    for (int j=0; j < model->ngeom; j++) {
      model->geom_friction[j] += friction_change;
      model->geom_friction[j] = std::max(model->geom_friction[j], 0.0);
    }

    std::cout << "Friction coefficients in model " << i << " boosted by " << friction_change << std::endl;
  }
}

}  // namespace mjpc
