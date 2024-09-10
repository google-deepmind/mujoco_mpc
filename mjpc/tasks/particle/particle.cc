// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/tasks/particle/particle.h"

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <string>
#include <vector>

#include "mjpc/states/state.h"
#include "mjpc/utilities.h"

namespace mjpc {

std::string Particle::XmlPath() const {
  return GetModelPath("particle/task_timevarying_task.xml");
}
std::string Particle::PlannerXmlPath() const {
  return GetModelPath("particle/task_timevarying_plan.xml");
}
std::string Particle::Name() const { return "Particle"; }

// -------- Residuals for particle task -------
//   Number of residuals: 3
//     Residual (0): position - goal_position
//     Residual (1): velocity
//     Residual (2): control
// --------------------------------------------
namespace {
void ResidualImpl(const mjModel* model, const mjData* data,
                  const double goal[2], double* residual) {
  // ----- residual (0) ----- //
  double* position = SensorByName(model, data, "position");
  mju_sub(residual, position, goal, model->nq);

  // ----- residual (1) ----- //
  double* velocity = SensorByName(model, data, "velocity");
  mju_copy(residual + 2, velocity, model->nv);

  // ----- residual (2) ----- //
  mju_copy(residual + 4, data->ctrl, model->nu);
}
}  // namespace

void Particle::ResidualFn::Residual(const mjModel* model, const mjData* data,
                                    double* residual) const {
  // some Lissajous curve
  double goal[2]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI)};
  ResidualImpl(model, data, goal, residual);
}

void Particle::TransitionLocked(mjModel* model, mjData* data) {
  // some Lissajous curve
  double goal[2]{0.25 * mju_sin(data->time), 0.25 * mju_cos(data->time / mjPI)};

  // update mocap position
  data->mocap_pos[0] = goal[0];
  data->mocap_pos[1] = goal[1];
}

void Particle::ModifyState(const mjModel* model, State* state) {
  // sampling token
  absl::BitGen gen_;

  // std from GUI
  double std_px = parameters[0];
  double std_py = parameters[1];
  double std_vx = parameters[2];
  double std_vy = parameters[3];

  // current state
  const std::vector<double>& s = state->state();

  // qpos
  double qpos[2]{s[0], s[1]};
  qpos[0] += absl::Gaussian<double>(gen_, 0.0, std_px);
  qpos[1] += absl::Gaussian<double>(gen_, 0.0, std_py);

  // qvel
  double qvel[2]{s[2], s[3]};
  qvel[0] += absl::Gaussian<double>(gen_, 0.0, std_vx);
  qvel[1] += absl::Gaussian<double>(gen_, 0.0, std_vy);

  // set state
  state->SetPosition(model, qpos);
  state->SetVelocity(model, qvel);
}

std::string ParticleFixed::XmlPath() const {
  return GetModelPath("particle/task_timevarying.xml");
}
std::string ParticleFixed::Name() const { return "ParticleFixed"; }

void ParticleFixed::ResidualFn::Residual(const mjModel* model,
                                         const mjData* data,
                                         double* residual) const {
  double goal[2]{data->mocap_pos[0], data->mocap_pos[1]};
  ResidualImpl(model, data, goal, residual);
}

}  // namespace mjpc
