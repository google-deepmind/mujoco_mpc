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

#include "mjpc/threadpool.h"

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>

namespace mjpc {
namespace {

// test threadpool scheduling
TEST(ThreadPoolTest, Count) {
  // pool
  ThreadPool pool(2);

  // count
  int count[3] = {0, 0, 0};

  // run
  {
    int count_before = pool.GetCount();
    for (int i = 0; i < 3; i++) {
      pool.Schedule([&count, i]() { count[i] += i; });
    }
    pool.WaitCount(count_before + 3);
  }
  pool.ResetCount();

  // test
  for (int i = 0; i < 3; i++) {
    EXPECT_EQ(count[i], i);
  }
}

}  // namespace
}  // namespace mjpc
