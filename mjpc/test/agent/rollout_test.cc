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

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/test/load.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

struct ParticleCopyTestTask : public mjpc::Task {
  std::string Name() const override {return ""; }
  std::string XmlPath() const override { return ""; }
  void Residual(const mjModel* model, const mjData* data,
                double* residual) const override {
    mju_copy(residual, data->qpos, model->nq);
    mju_copy(residual + model->nq, data->qvel, model->nv);
  }
};

ParticleCopyTestTask task;

extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    task.Residual(model, data, data->sensordata);
  }
}

// test trajectory rollout with PD controller on particle task
TEST(RolloutTest, Particle) {
  // load model
  mjModel* model = LoadTestModel("particle_task.xml");
  task.Reset(model);

  // create data
  mjData* data = mj_makeData(model);

  ASSERT_EQ(model->nq + model->nv, task.num_residual);
  int num_residual = task.num_residual;

  // set callback
  mjcb_sensor = sensor;

  // set data
  mj_forward(model, data);

  // policy
  double position_goal[2] = {0.1, 0.1};
  double velocity_goal[2] = {0.0, 0.0};
  auto feedback_policy = [&position_goal, &velocity_goal](
                             double* action, const double* state, double time) {
    // goal error
    double position_error[2];
    mju_sub(position_error, state, position_goal, 2);

    // velocity error
    double velocity_error[2];
    mju_sub(velocity_error, state + 2, velocity_goal, 2);

    // feedback gains
    double P = 10.0;
    double D = 2.5;

    // action
    mju_scl(action, position_error, -P, 2);
    mju_addToScl(action, velocity_error, -D, 2);
  };

  // trajectory
  Trajectory trajectory;
  int horizon = 100;
  trajectory.Initialize(model->nq + model->nv, model->nu, num_residual,
                        1, horizon);
  trajectory.Allocate(horizon);

  // ----- rollout ----- //

  // initial state
  double state[4] = {0.0, 0.0, 0.0, 0.0};
  double time = 0.0;
  double mocap[7];
  mju_copy(mocap, data->mocap_pos, 3);
  mju_copy(mocap + 3, data->mocap_quat, 4);

  // rollout feedback policy
  trajectory.Rollout(feedback_policy, &task, model, data, state, time, mocap,
                     NULL, horizon);

  // test final state
  double position_error[2];
  double velocity_error[2];
  mju_sub(position_error,
          DataAt(trajectory.states, (horizon - 1) * (model->nq + model->nv)),
          position_goal, 2);
  mju_sub(velocity_error,
          DataAt(trajectory.states,
                 (horizon - 1) * (model->nq + model->nv) + model->nq),
          velocity_goal, 2);

  EXPECT_NEAR(mju_L1(position_error, 2), 0.0, 0.1);
  EXPECT_NEAR(mju_L1(velocity_error, 2), 0.0, 0.1);

  // test residual
  double residual_error[400];
  mju_sub(residual_error, trajectory.states.data(), trajectory.residual.data(),
          400);

  EXPECT_NEAR(mju_L1(residual_error, 400), 0.0, 1.0e-5);

  // delete model + data
  mj_deleteData(data);
  mj_deleteModel(model);

  // unset callback
  mjcb_sensor = nullptr;
}

}  // namespace
}  // namespace mjpc
