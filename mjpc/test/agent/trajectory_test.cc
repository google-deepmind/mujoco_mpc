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

#include "mjpc/trajectory.h"

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>

namespace mjpc {
namespace {

// test trajectory reset
TEST(TrajectoryTest, Reset) {
  // trajectory
  Trajectory trajectory;

  // allocate
  trajectory.Initialize(2, 2, 2, 1, 2);
  trajectory.Allocate(2);

  // fill
  mju_fill(trajectory.states.data(), 1.0, 2 * 2);
  mju_fill(trajectory.actions.data(), 1.0, 2 * 2);
  mju_fill(trajectory.times.data(), 1.0, 2);
  mju_fill(trajectory.residual.data(), 1.0, 2 * 2);
  mju_fill(trajectory.costs.data(), 1.0, 2);
  trajectory.total_return = 1.0;
  mju_fill(trajectory.trace.data(), 1.0, 3 * 2);

  // reset
  trajectory.Reset(2);

  // test
  EXPECT_NEAR(mju_norm(trajectory.states.data(), 2 * 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(trajectory.actions.data(), 2 * 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(trajectory.times.data(), 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(trajectory.residual.data(), 2 * 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(trajectory.costs.data(), 2), 0.0, 1.0e-5);
  EXPECT_NEAR(trajectory.total_return, 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(trajectory.trace.data(), 3 * 2), 0.0, 1.0e-5);
}

// test trajectory copy
TEST(TrajectoryTest, Copy) {
  // trajectory
  Trajectory trajectory;
  Trajectory trajectory_copy;

  // allocate
  trajectory.Initialize(2, 2, 2, 1, 2);
  trajectory.Allocate(2);
  trajectory_copy.Initialize(2, 2, 2, 1, 2);
  trajectory_copy.Allocate(2);

  // fill
  mju_fill(trajectory.states.data(), 1.0, 2 * 2);
  mju_fill(trajectory.actions.data(), 1.0, 2 * 2);
  mju_fill(trajectory.times.data(), 1.0, 2);
  mju_fill(trajectory.residual.data(), 1.0, 2 * 2);
  mju_fill(trajectory.costs.data(), 1.0, 2);
  trajectory.total_return = 1.0;
  mju_fill(trajectory.trace.data(), 1.0, 3 * 2);

  // copy
  trajectory_copy = trajectory;

  // trajectory differences
  double state_error[4];
  double action_error[4];
  double times_error[2];
  double residual_error[4];
  double costs_error[2];
  double trace_error[6];

  mju_sub(state_error, trajectory.states.data(), trajectory_copy.states.data(),
          4);
  mju_sub(action_error, trajectory.actions.data(),
          trajectory_copy.actions.data(), 4);
  mju_sub(times_error, trajectory.times.data(), trajectory_copy.times.data(),
          2);
  mju_sub(residual_error, trajectory.residual.data(),
          trajectory_copy.residual.data(), 4);
  mju_sub(costs_error, trajectory.costs.data(), trajectory_copy.costs.data(),
          2);
  mju_sub(trace_error, trajectory.trace.data(), trajectory_copy.trace.data(),
          6);

  EXPECT_NEAR(mju_norm(state_error, 2 * 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(action_error, 2 * 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(times_error, 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(residual_error, 2 * 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(costs_error, 2), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(trace_error, 3 * 2), 0.0, 1.0e-5);
  EXPECT_EQ(trajectory.horizon - trajectory_copy.horizon, 0);
  EXPECT_NEAR(mju_abs(trajectory.total_return - trajectory_copy.total_return),
              0.0, 1.0e-5);
}

}  // namespace
}  // namespace mjpc
