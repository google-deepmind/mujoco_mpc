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

#include "mjpc/states/state.h"

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/test/load.h"
#include "mjpc/utilities.h"

namespace mjpc {
class StateTest : public ::testing::Test {
 protected:
  // test State class
  void TestState() {
    // load model
    mjModel* model = LoadTestModel("particle_task.xml");

    // create data
    mjData* data = mj_makeData(model);

    // set data
    mj_forward(model, data);

    // state
    State state;

    // allocate
    state.Allocate(model);

    // set data state
    mju_fill(data->qpos, 1.0, model->nq);
    mju_fill(data->qvel, 2.0, model->nv);

    // set mocap state
    double mocap[7] = {1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0};
    mju_copy(data->mocap_pos, mocap, 3);
    mju_copy(data->mocap_quat, mocap + 3, 4);

    // set state state
    state.Set(model, data);

    // test state state
    EXPECT_NEAR(state.state_[0], 1.0, 1.0e-5);
    EXPECT_NEAR(state.state_[1], 1.0, 1.0e-5);
    EXPECT_NEAR(state.state_[2], 2.0, 1.0e-5);
    EXPECT_NEAR(state.state_[2], 2.0, 1.0e-5);

    // test state mocap state
    double mocap_error[7];
    mju_sub(mocap_error, state.mocap_.data(), data->mocap_pos, 3);
    mju_sub(mocap_error + 3, DataAt(state.mocap_, 3), data->mocap_quat, 4);

    EXPECT_NEAR(mju_L1(mocap_error, 7), 0.0, 1.0e-5);

    // reset
    state.Reset();

    // test reset
    EXPECT_NEAR(mju_L1(state.state_.data(), 4), 0.0, 1.0e-5);
    EXPECT_NEAR(mju_L1(state.mocap_.data(), 7), 0.0, 1.0e-5);

    // delete model + data
    mj_deleteData(data);
    mj_deleteModel(model);
  }
};

namespace {
TEST_F(StateTest, State) { TestState(); }
}  // namespace
}  // namespace mjpc
