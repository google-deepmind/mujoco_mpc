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

}  // namespace mjpc
