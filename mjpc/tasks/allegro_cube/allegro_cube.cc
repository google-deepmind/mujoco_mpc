#include "mjpc/tasks/allegro_cube/allegro_cube.h"

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

// ------- Residuals for cartpole task ------
//     Vertical: Pole angle cosine should be -1
//     Centered: Cart should be at goal position
//     Velocity: Pole angular velocity should be small
//     Control:  Control should be small
// ------------------------------------------
void AllegroCube::ResidualFn::Residual(const mjModel* model, const mjData* data,
                        double* residual) const {
  // ---------- Vertical ----------
  residual[0] = std::cos(data->qpos[1]) - 1;

  // ---------- Centered ----------
  residual[1] = data->qpos[0] - parameters_[0];

  // ---------- Velocity ----------
  residual[2] = data->qvel[1];

  // ---------- Control ----------
  residual[3] = data->ctrl[0];
}

}  // namespace mjpc
