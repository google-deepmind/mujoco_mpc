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

#include "mjpc/utilities.h"

#include <atomic>
#include <cstdint>
#include <limits>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/test/load.h"
#include "mjpc/threadpool.h"

namespace mjpc {
namespace {
using ::testing::IsNull;

TEST(UtilitiesTest, State) {
  // load model
  mjModel* model = LoadTestModel("particle_task.xml");

  // create data
  mjData* data = mj_makeData(model);

  // set data
  mj_forward(model, data);

  // state
  const double state[4] = {0.1, 0.2, 0.3, 0.4};

  // set state
  SetState(model, data, state);

  // test data state
  EXPECT_NEAR(data->qpos[0], state[0], 1.0e-5);
  EXPECT_NEAR(data->qpos[1], state[1], 1.0e-5);
  EXPECT_NEAR(data->qvel[0], state[2], 1.0e-5);
  EXPECT_NEAR(data->qvel[1], state[3], 1.0e-5);

  // new state
  double new_state[4];
  GetState(model, data, new_state);

  // test new state
  EXPECT_NEAR(new_state[0], state[0], 1.0e-5);
  EXPECT_NEAR(new_state[1], state[1], 1.0e-5);
  EXPECT_NEAR(new_state[2], state[2], 1.0e-5);
  EXPECT_NEAR(new_state[3], state[3], 1.0e-5);

  // delete model + data
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(UtilitiesTest, CustomNumeric) {
  // load model
  mjModel* model = LoadTestModel("particle_task.xml");

  // get invalid from custom numeric
  {
    double* const x = GetCustomNumericData(model, "invalid");
    EXPECT_THAT(x, IsNull());
  }

  // get number from custom numeric
  {
    const double x = GetCustomNumericData(model, "test_double")[0];
    EXPECT_NEAR(x, 0.1, 1.0e-5);
  }

  // set double from custom numeric
  {
    const double x = [&]() -> double {
      const auto maybe_x = GetNumber<double>(model, "test_double");
      EXPECT_TRUE(maybe_x.has_value());
      return *maybe_x;
    }();
    EXPECT_NEAR(x, 0.1, 1.0e-5);
  }
  {
    const auto x = GetNumberOrDefault(
        std::numeric_limits<double>::quiet_NaN(), model, "test_double");
    static_assert(std::is_same_v<decltype(x), const double>);
    EXPECT_NEAR(x, 0.1, 1.0e-5);
  }

  // set double from invalid numeric
  {
    const auto x = GetNumberOrDefault(16.0, model, "invalid");
    static_assert(std::is_same_v<decltype(x), const double>);
    EXPECT_EQ(x, 16.0);
  }

  // set int from custom numeric
  {
    const int x = [&]() -> int {
      const auto maybe_x = GetNumber<int>(model, "test_int");
      EXPECT_TRUE(maybe_x.has_value());
      return *maybe_x;
    }();
    EXPECT_EQ(x, 1);
  }
  {
    const auto x =
        GetNumberOrDefault(std::numeric_limits<int>::max(), model, "test_int");
    static_assert(std::is_same_v<decltype(x), const int>);
    EXPECT_EQ(x, 1);
  }

  // set int from invalid numeric
  {
    const auto x = GetNumberOrDefault(16, model, "invalid");
    static_assert(std::is_same_v<decltype(x), const int>);
    EXPECT_EQ(x, 16);
  }

  enum class TestEnum : std::uint64_t {
    kFirstValue,
    kSecondValue,
    kThirdValue,
  };

  // set enum from numeric
  {
    const auto x = GetNumberOrDefault(TestEnum::kFirstValue, model, "test_int");
    static_assert(std::is_same_v<decltype(x), const TestEnum>);
    EXPECT_EQ(x, TestEnum::kSecondValue);
  }
  {
    const auto x = GetNumberOrDefault(TestEnum::kFirstValue, model, "invalid");
    static_assert(std::is_same_v<decltype(x), const TestEnum>);
    EXPECT_EQ(x, TestEnum::kFirstValue);
  }

  // delete model
  mj_deleteModel(model);
}

TEST(UtilitiesTest, ByName) {
  // load model
  mjModel* model = LoadTestModel("particle_task.xml");

  // create data
  mjData* data = mj_makeData(model);

  // set data
  mj_forward(model, data);

  // set sensor value ("Position")
  data->sensordata[0] = 1.0;

  // set sensor value ("Velocity")
  data->sensordata[2] = 2.0;

  // get sensor value by name ("Position")
  double* position = SensorByName(model, data, "Position");

  // get sensor value by name ("Velocity")
  double* velocity = SensorByName(model, data, "Velocity");

  EXPECT_NEAR(position[0], 1.0, 1.0e-5);
  EXPECT_NEAR(velocity[0], 2.0, 1.0e-5);

  // get invalid key qpos
  double* qpos_invalid;
  qpos_invalid = KeyQPosByName(model, data, "invalid");

  EXPECT_EQ(qpos_invalid, nullptr);

  // get "home" key qpos
  double* qpos_home;
  qpos_home = KeyQPosByName(model, data, "home");

  EXPECT_NEAR(qpos_home[0], 1.0, 1.0e-5);
  EXPECT_NEAR(qpos_home[1], 2.0, 1.0e-5);

  // get invalid key qvel
  double* qvel_invalid;
  qvel_invalid = KeyQVelByName(model, data, "invalid");

  EXPECT_EQ(qvel_invalid, nullptr);

  // get "home" key qvel
  double* qvel_home;
  qvel_home = KeyQVelByName(model, data, "home");

  EXPECT_NEAR(qvel_home[0], -1.0, 1.0e-5);
  EXPECT_NEAR(qvel_home[1], -2.0, 1.0e-5);

  // delete model + data
  mj_deleteData(data);
  mj_deleteModel(model);
}

TEST(UtilitiesTest, Clamp) {
  // bounds
  double bounds[6] = {-1.0, 1.0, -1.0, 1.0, -1.0, 1.0};

  // value
  double x[3] = {-2.0, 3.0, 0.0};

  // Clamp
  Clamp(x, bounds, 3);

  EXPECT_NEAR(x[0], -1.0, 1.0e-5);
  EXPECT_NEAR(x[1], 1.0, 1.0e-5);
  EXPECT_NEAR(x[2], 0.0, 1.0e-5);
}

TEST(UtilitiesTest, PowerSequence) {
  // sequence
  double sequence[4] = {0.2, 0.3, 0.4, 0.5};
  int length = 4;
  double step = 0.1;

  // power
  double power = 2.0;

  // power sequence
  PowerSequence(sequence, step, sequence[0], sequence[3], power, length);

  // test
  EXPECT_NEAR(sequence[0], 0.2, 1.0e-5);
  EXPECT_NEAR(sequence[1], 0.27142857, 1.0e-5);
  EXPECT_NEAR(sequence[2], 0.37142857, 1.0e-5);
  EXPECT_NEAR(sequence[3], 0.5, 1.0e-5);
}

TEST(UtilitiesTest, FindInterval) {
  // sequence
  std::vector<double> sequence{-1.0, 0.0, 1.0, 2.0};
  int length = 4;

  // bounds
  int bounds[2];

  // get internal interval
  FindInterval(bounds, sequence, 0.5, length);

  EXPECT_EQ(bounds[0], 1);
  EXPECT_EQ(bounds[1], 2);

  // get lower interval
  FindInterval(bounds, sequence, -2.0, length);

  EXPECT_EQ(bounds[0], 0);
  EXPECT_EQ(bounds[1], 0);

  // get lower interval
  FindInterval(bounds, sequence, 2.1, length);

  EXPECT_EQ(bounds[0], length - 1);
  EXPECT_EQ(bounds[1], length - 1);
}

TEST(UtilitiesTest, LinearInterpolation) {
  // x
  std::vector<double> x{1.0, 2.0};
  double y[2] = {1.0, 2.0};

  // inside
  double y1;
  LinearInterpolation(&y1, 1.5, x, y, 1, 2);
  EXPECT_NEAR(y1, 1.5, 1.0e-5);

  // lower extrapolation
  double y2;
  LinearInterpolation(&y2, 0.5, x, y, 1, 2);
  EXPECT_NEAR(y2, 1.0, 1.0e-5);

  // upper extrapolation
  double y3;
  LinearInterpolation(&y3, 2.5, x, y, 1, 2);
  EXPECT_NEAR(y3, 2.0, 1.0e-5);
}

TEST(UtilitiesTest, IncrementAtomic) {
  const double kInitial = 1;
  std::atomic<double> v = kInitial;
  const int kN = 10;
  {
    mjpc::ThreadPool pool(kN);
    for (int i = 0; i < kN; ++i) {
      pool.Schedule([&v]() {
        IncrementAtomic(v, 10);
      });
    }
  }
  EXPECT_EQ(v.load(), kInitial + 10 * kN);
}

}  // namespace
}  // namespace mjpc
