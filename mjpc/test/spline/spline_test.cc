// Copyright 2024 DeepMind Technologies Limited
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

#include "mjpc/spline/spline.h"
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <absl/types/span.h>

namespace {
using mjpc::spline::TimeSpline;
using ::testing::ElementsAre;
using ::testing::TestWithParam;

struct TimeSplineTestCase {
  std::string test_name;
  int reserve = 0;
};

using TimeSplineReserveTest = TestWithParam<TimeSplineTestCase>;

TEST(TimeSplineTest, Empty) {
  TimeSpline spline(/*dim=*/10);
  EXPECT_EQ(spline.Size(), 0);
  EXPECT_EQ(spline.Dim(), 10);
  std::vector<double> values(10, 1.0);
  spline.Sample(2.0, absl::MakeSpan(values));
  for (double v : values) {
    EXPECT_EQ(v, 0.0);
  }
}

TEST(TimeSplineTest, OneNode) {
  TimeSpline spline(/*dim=*/2);

  spline.AddNode(1.0, {1.0, 2.0});
  EXPECT_EQ(spline.Size(), 1);
  for (double time : {0.0, 2.0, 4.0}) {
    EXPECT_THAT(spline.Sample(time), ElementsAre(1.0, 2.0));
  }
}

TEST(TimeSplineTest, TwoNodes) {
  TimeSpline spline(/*dim=*/2);

  spline.AddNode(1.0, {1.0, 2.0});
  // an alternative method of adding a node: setting values after AddNode
  const TimeSpline::Node n = spline.AddNode(2.0);
  n.values()[0] = 3.0;
  n.values()[1] = 4.0;
  EXPECT_EQ(spline.Size(), 2);

  EXPECT_THAT(spline.Sample(0), ElementsAre(1.0, 2.0));
  EXPECT_THAT(spline.Sample(1), ElementsAre(1.0, 2.0));
  EXPECT_THAT(spline.Sample(2), ElementsAre(3.0, 4.0));
  EXPECT_THAT(spline.Sample(3), ElementsAre(3.0, 4.0));
}

TEST_P(TimeSplineReserveTest, AddNodeBeforeStart) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.Reserve(test_case.reserve);

  spline.AddNode(2.0, {2.0, 3.0});
  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(3.0, {3.0, 4.0});
  spline.AddNode(0.0, {0.0, 1.0});

  EXPECT_THAT(spline.Sample(0), ElementsAre(0.0, 1.0));
  EXPECT_THAT(spline.Sample(1), ElementsAre(1.0, 2.0));
  EXPECT_THAT(spline.Sample(2), ElementsAre(2.0, 3.0));
  EXPECT_THAT(spline.Sample(3), ElementsAre(3.0, 4.0));
}

TEST(TimeSplineTest, AddNodeResetsToZeroByDefault) {
  TimeSpline spline(/*dim=*/1);
  // Make sure the values_ array is nonzero
  spline.Reserve(6);
  for (int i = 0; i < 6; ++i) {
    spline.AddNode(2.0 * i, {1.0});
  }

  // Clear the spline (shouldn't change values_ array).
  spline.Clear();

  // When a node is added, the default values should be 0.
  spline.AddNode(1.0);

  EXPECT_THAT(spline.Sample(0), ElementsAre(0.0));
}

TEST(TimeSplineTest, ZeroOrder) {
  TimeSpline spline(/*dim=*/2);
  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {3.0, 4.0});

  EXPECT_THAT(spline.Sample(1.5), ElementsAre(1.0, 2.0));
}

TEST(TimeSplineTest, DiscardBefore) {
  TimeSpline spline(/*dim=*/2);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});
  spline.AddNode(3.0, {3.0, 4.0});
  spline.AddNode(4.0, {4.0, 5.0});
  EXPECT_EQ(spline.Size(), 4);

  EXPECT_THAT(spline.Sample(1.0), ElementsAre(1.0, 2.0));

  // Calling DiscardBefore(0.9) should have no effect, since there are no
  // nodes before that time anyway.
  EXPECT_EQ(spline.DiscardBefore(0.9), 0);
  EXPECT_EQ(spline.Size(), 4);
  EXPECT_THAT(spline.Sample(0.0), ElementsAre(1.0, 2.0));

  // Once the first two values are discarded, early time values should get the
  // {3.0, 4.0} value.
  int discarded = spline.DiscardBefore(3.0);
    EXPECT_EQ(discarded, 2);
    EXPECT_EQ(spline.Size(), 2);
    EXPECT_THAT(spline.Sample(1.0), ElementsAre(3.0, 4.0));

  // Discarding just after 3.0 should have no effect.
  EXPECT_EQ(spline.DiscardBefore(3.9), 0);
    EXPECT_EQ(spline.Size(), 2);
    EXPECT_THAT(spline.Sample(1.0), ElementsAre(3.0, 4.0));
}

TEST(TimeSplineTest, DiscardBeforeRingLoop) {
  TimeSpline spline(/*dim=*/1);
  spline.Reserve(4);

  spline.AddNode(1.0, {1.0});
  spline.AddNode(2.0, {2.0});
  spline.AddNode(3.0, {3.0});
  spline.AddNode(4.0, {4.0});
  EXPECT_EQ(spline.Size(), 4);

  EXPECT_EQ(spline.DiscardBefore(3), 2);
  EXPECT_EQ(spline.Size(), 2);
  // At this point, values_begin_ = 2, values_end_ = 4. Adding two more entries
  // so the ring buffer loops around.
  spline.AddNode(5.0, {5.0});
  spline.AddNode(6.0, {6.0});

  // Remove elements so that values_begin_ has to go around the end of the
  // buffer.
  EXPECT_EQ(spline.DiscardBefore(6.0), 3);
  EXPECT_EQ(spline.Size(), 1);

  EXPECT_EQ(spline.Sample(1.0)[0], 6.0);
}

TEST(TimeSplineTest, ReserveAfterAdd) {
  TimeSpline spline(/*dim=*/2);
  spline.Reserve(3);
  EXPECT_EQ(spline.Size(), 0);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});

  spline.Reserve(6);
  spline.AddNode(3.0, {4.0, 5.0});
  EXPECT_EQ(spline.Size(), 3);

  EXPECT_THAT(spline.Sample(2.5), ElementsAre(2.0, 3.0));
}

TEST_P(TimeSplineReserveTest, MoveConstructor) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.Reserve(test_case.reserve);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});
  spline.AddNode(3.0, {3.0, 4.0});
  spline.AddNode(4.0, {4.0, 5.0});

  TimeSpline spline2(std::move(spline));
  EXPECT_EQ(spline2.Size(), 4);
  EXPECT_THAT(spline2.Sample(2.5), ElementsAre(2.0, 3.0));
}

TEST_P(TimeSplineReserveTest, MoveAssignment) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.Reserve(test_case.reserve);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});
  spline.AddNode(3.0, {3.0, 4.0});
  spline.AddNode(4.0, {4.0, 5.0});

  TimeSpline spline2(/*dim=*/3);
  // Set some properties that should be overridden by assignment.
  spline2.Reserve(7);
  spline2.AddNode(2.5, {5.0, 6.0, 7.0});

  spline2 = std::move(spline);
  EXPECT_EQ(spline2.Dim(), 2);
  EXPECT_EQ(spline2.Size(), 4);
  EXPECT_THAT(spline2.Sample(2.5), ElementsAre(2.0, 3.0));
}

TEST_P(TimeSplineReserveTest, CopyConstructor) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.Reserve(test_case.reserve);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});
  spline.DiscardBefore(2.0);
  spline.AddNode(3.0, {3.0, 4.0});
  spline.AddNode(4.0, {4.0, 5.0});
  spline.AddNode(5.0, {5.0, 6.0});

  TimeSpline spline2(spline);
  EXPECT_EQ(spline2.Size(), 4);

  // clear original spline
  spline.Clear();

  // spline2 should be unaffected
  EXPECT_THAT(spline2.Sample(1.5), ElementsAre(2.0, 3.0));
  EXPECT_THAT(spline2.Sample(2.5), ElementsAre(2.0, 3.0));
}

TEST_P(TimeSplineReserveTest, CopyAssignment) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.Reserve(test_case.reserve);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});
  spline.DiscardBefore(2.0);
  spline.AddNode(3.0, {3.0, 4.0});
  spline.AddNode(4.0, {4.0, 5.0});
  spline.AddNode(5.0, {5.0, 6.0});

  TimeSpline spline2(/*dim=*/3);
  // Set some properties that should be overridden by assignment.
  spline2.Reserve(7);
  spline2.AddNode(2.5, {5.0, 6.0, 7.0});

  spline2 = spline;
  EXPECT_EQ(spline2.Size(), 4);

  // clear original spline
  spline.Clear();

  // spline2 should be unaffected
  EXPECT_THAT(spline2.Sample(1.5), ElementsAre(2.0, 3.0));
  EXPECT_THAT(spline2.Sample(2.5), ElementsAre(2.0, 3.0));
}

TEST_P(TimeSplineReserveTest, Clear) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.Reserve(test_case.reserve);

  spline.AddNode(1.0, {1.0, 2.0});
  EXPECT_THAT(spline.Sample(0), ElementsAre(1.0, 2.0));

  spline.Clear();
  EXPECT_EQ(spline.Size(), 0);
  EXPECT_THAT(spline.Sample(0), ElementsAre(0.0, 0.0));

  spline.AddNode(1.0, {3.0, 4.0});
  EXPECT_THAT(spline.Sample(1), ElementsAre(3.0, 4.0));
}

TEST(TimeSplineTest, Dim0) {
  // Degenerate case of a spline with no values.
  TimeSpline spline(/*dim=*/0);
  spline.Reserve(10);

  spline.AddNode(1.0, {});
  spline.AddNode(2.0, {});
  EXPECT_EQ(spline.Size(), 2);
  EXPECT_EQ(spline.DiscardBefore(2.0), 1);
  EXPECT_EQ(spline.Size(), 1);
  std::vector<double> values;
  spline.Sample(1, absl::MakeSpan(values.data(), values.size()));
}

INSTANTIATE_TEST_SUITE_P(
    TimeSplineReserve, TimeSplineReserveTest,
    testing::ValuesIn<TimeSplineTestCase>({
        {"Reserve0", 0},
        {"Reserve4", 4},
        {"Reserve7", 7},
    }),
    [](const testing::TestParamInfo<TimeSplineReserveTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
