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

#include <string>

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/planners/gradient/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/test/load.h"
#include "mjpc/test/testdata/particle_residual.h"
#include "mjpc/threadpool.h"

namespace mjpc {
namespace {

// model
mjModel* model;

// state
State state;

ParticleTestTask task;

// sensor
extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    task.Residual(model, data, data->sensordata);
  }
}

// test gradient planner on particle task
TEST(GradientPlannerTest, Particle) {
  // load model
  model = LoadTestModel("particle_task.xml");
  task.Reset(model);

  // create data
  mjData* data = mj_makeData(model);

  // set data
  mj_forward(model, data);

  // ----- state ----- //
  state.Initialize(model);
  state.Allocate(model);
  state.Reset();
  state.Set(model, data);

  // ----- gradient planner ----- //
  GradientPlanner planner;
  planner.Initialize(model, task);
  planner.Allocate();
  planner.Reset(kMaxTrajectoryHorizon);

  // ----- settings ----- //
  int iterations = 50;
  double horizon = 2.5;
  double timestep = 0.1;
  int steps =
      mju_max(mju_min(horizon / timestep + 1, kMaxTrajectoryHorizon), 1);
  model->opt.timestep = timestep;

  // sensor callback
  mjcb_sensor = sensor;

  // threadpool
  ThreadPool pool(1);

  // set state
  planner.SetState(state);

  // ---- optimize ----- //
  for (int i = 0; i < iterations; i++) {
    planner.OptimizePolicy(steps, pool);
  }

  // test final state
  EXPECT_NEAR(planner.trajectory[planner.winner].states[(steps - 1) * (model->nq + model->nv)],
              state.mocap()[0], 1.0e-2);
  EXPECT_NEAR(planner.trajectory[planner.winner].states[(steps - 1) * (model->nq + model->nv) + 1],
              state.mocap()[1], 1.0e-2);
  EXPECT_NEAR(planner.trajectory[planner.winner].states[(steps - 1) * (model->nq + model->nv) + 2],
              0.0, 1.0e-1);
  EXPECT_NEAR(planner.trajectory[planner.winner].states[(steps - 1) * (model->nq + model->nv) + 3],
              0.0, 1.0e-1);

  // test action limits
  for (int t = 0; t < steps - 1; t++) {
    for (int i = 0; i < model->nu; i++) {
      EXPECT_LE(
          planner.trajectory[planner.winner].actions[t * model->nu + i],
          model->actuator_ctrlrange[2 * i + 1]);
      EXPECT_GE(
          planner.trajectory[planner.winner].actions[t * model->nu + i],
          model->actuator_ctrlrange[2 * i]);
    }
  }

  // delete data
  mj_deleteData(data);

  // delete model
  mj_deleteModel(model);
}

}  // namespace
}  // namespace mjpc
