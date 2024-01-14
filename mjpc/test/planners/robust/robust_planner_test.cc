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

#include <memory>

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/planners/robust/robust_planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/test/load.h"
#include "mjpc/test/testdata/particle_residual.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"

namespace mjpc {
namespace {

// load model
mjModel* model;

// state
State state;

// task
ParticleTestTask task;

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    task.Residual(model, data, data->sensordata);
  }
}

// test robust sampling planner on particle task
TEST(RobustPlannerTest, RandomSearch) {
  // load model
  model = LoadTestModel("particle_task.xml");
  task.Reset(model);

  // create data
  mjData* data = mj_makeData(model);
  // the "home" keyframe initializes the state too far from the target
  int home_id = mj_name2id(model, mjOBJ_KEY, "ctrl_test");
  mj_resetDataKeyframe(model, data, home_id);

  // ----- state ----- //
  state.Initialize(model);
  state.Allocate(model);
  state.Reset();
  state.Set(model, data);

  // ----- sampling planner ----- //
  RobustPlanner planner(std::make_unique<SamplingPlanner>());
  planner.Initialize(model, task);
  planner.Allocate();
  // If there's no keyframe, data->ctrl will be zeros, so this is always safe.
  planner.Reset(kMaxTrajectoryHorizon, data->ctrl);

  double res[2];
  // look at some arbitrary, hard-coded time:
  planner.ActionFromPolicy(res, state.state().data(), 2);
  // expected values copied from the keyframe in the xml:
  EXPECT_NEAR(res[0], 0.1, 1.0e-4);
  EXPECT_NEAR(res[1], 0.2, 1.0e-4);

  // ----- settings ----- //
  int iterations = 1000;
  double horizon = 2.5;
  double timestep = 0.1;
  int steps =
      mju_max(mju_min(horizon / timestep + 1, kMaxTrajectoryHorizon), 1);
  model->opt.timestep = timestep;

  // sensor callback
  mjcb_sensor = sensor;

  // threadpool
  ThreadPool pool(1);

  // ----- initial state ----- //
  planner.SetState(state);

  // ---- optimize w/ random search ----- //
  for (int i = 0; i < iterations; i++) {
    planner.OptimizePolicy(steps, pool);
  }

  // test final state
  int final_state_index = (steps - 1) * (model->nq + model->nv);
  ASSERT_GE(planner.BestTrajectory()->states.size(), final_state_index);
  EXPECT_NEAR(planner.BestTrajectory()->states[final_state_index],
              state.mocap()[0], 1.0e-1);
  EXPECT_NEAR(planner.BestTrajectory()->states[final_state_index + 1],
              state.mocap()[1], 1.0e-1);
  EXPECT_NEAR(planner.BestTrajectory()->states[final_state_index + 2], 0.0,
              1.0e-1);
  EXPECT_NEAR(planner.BestTrajectory()->states[final_state_index + 3], 0.0,
              1.0e-1);

  // test action limits
  for (int t = 0; t < steps - 1; t++) {
    for (int i = 0; i < model->nu; i++) {
      EXPECT_LE(planner.BestTrajectory()->actions[t * model->nu + i],
                model->actuator_ctrlrange[2 * i + 1]);
      EXPECT_GE(planner.BestTrajectory()->actions[t * model->nu + i],
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
