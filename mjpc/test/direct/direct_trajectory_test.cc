// Copyright 2023 DeepMind Technologies Limited
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

#include <vector>

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include "gtest/gtest.h"
#include "mjpc/direct/trajectory.h"

namespace mjpc {
namespace {

TEST(DirectTrajectory, Test) {
  // dimensions
  int dim = 2;
  int length = 3;

  // trajectory
  DirectTrajectory<double> trajectory;

  // initialize
  trajectory.Initialize(dim, length);

  // test initialization
  EXPECT_EQ(trajectory.Head(), 0);
  EXPECT_NEAR(mju_norm(trajectory.Data(), trajectory.Length()), 0.0, 1.0e-5);
  EXPECT_NEAR(mju_norm(trajectory.Data(), trajectory.Length()), 0.0, 1.0e-5);

  // random data
  std::vector<double> data(dim * length);
  absl::BitGen gen_;
  for (int i = 0; i < dim * length; i++) {
    data[i] = absl::Gaussian<double>(gen_, 0.0, 1.0);
  }

  // set data
  for (int i = 0; i < length; i++) {
    double* element = data.data() + dim * i;
    trajectory.Set(element, i);
  }

  // test get data
  std::vector<double> error(dim);
  for (int i = 0; i < length; i++) {
    // get elements
    double* element = data.data() + dim * i;
    double* traj_element = trajectory.Get(i);

    // error
    mju_sub(error.data(), traj_element, element, dim);

    // test
    EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);
  }

  // elements
  double* e0 = data.data() + dim * 0;
  double* e1 = data.data() + dim * 1;
  double* e2 = data.data() + dim * 2;

  // shift head index
  trajectory.Shift(1);

  // trajectory elements
  double* t0 = trajectory.Get(0);
  double* t1 = trajectory.Get(1);
  double* t2 = trajectory.Get(2);

  // t0 - e1
  mju_sub(error.data(), t0, e1, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // t1 - e2
  mju_sub(error.data(), t1, e2, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // t2 - e0
  mju_sub(error.data(), t2, e0, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // test set
  double s[2] = {1.1, 3.24};
  trajectory.Set(s, 0);

  // t0 - s
  mju_sub(error.data(), t0, s, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // data_ + dim * 1 - s
  mju_sub(error.data(), trajectory.Data() + dim * 1, s, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // shift head index
  trajectory.Shift(1);

  // trajectory elements
  t0 = trajectory.Get(0);
  t1 = trajectory.Get(1);
  t2 = trajectory.Get(2);

  // t0 - e2
  mju_sub(error.data(), t0, e2, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // t1 - e0
  mju_sub(error.data(), t1, e0, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // t2 - s
  mju_sub(error.data(), t2, s, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // data_ + dim * 1 - s
  mju_sub(error.data(), trajectory.Data() + dim * 1, s, dim);
  EXPECT_NEAR(mju_norm(error.data(), dim), 0.0, 1.0e-4);

  // ----- shift head ----- //
  trajectory.ResetHead();

  // shift by 1
  trajectory.Shift(1);
  EXPECT_EQ(trajectory.Head(), 1);

  // shift by 1
  trajectory.Shift(1);
  EXPECT_EQ(trajectory.Head(), 2);

  // shift by 1
  trajectory.Shift(1);
  EXPECT_EQ(trajectory.Head(), 0);

  // shift by 3
  trajectory.Shift(trajectory.Length());
  EXPECT_EQ(trajectory.Head(), 0);

  // shift by 2
  trajectory.Shift(2);
  EXPECT_EQ(trajectory.Head(), 2);

  // shift by length
  trajectory.Shift(length);
  EXPECT_EQ(trajectory.Head(), 2);

  // shift by 2 * length
  trajectory.Shift(2 * length);
  EXPECT_EQ(trajectory.Head(), 2);

  // shift by 2 * length + 1
  trajectory.Shift(2 * length + 1);
  EXPECT_EQ(trajectory.Head(), 0);
}

}  // namespace
}  // namespace mjpc
