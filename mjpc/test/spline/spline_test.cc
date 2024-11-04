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
#include <cmath>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <absl/types/span.h>

namespace {
using mjpc::spline::TimeSpline;
using mjpc::spline::SplineInterpolation;
using ::testing::ElementsAre;
using ::testing::TestWithParam;

struct TimeSplineTestCase {
  std::string test_name;
  SplineInterpolation interpolation;
  int reserve = 0;
};

using TimeSplineAllInterpolationsTest = TestWithParam<TimeSplineTestCase>;
using TimeSplineReserveTest = TestWithParam<TimeSplineTestCase>;

TEST(TimeSplineAllInterpolationsTest, Empty) {
  TimeSpline spline(/*dim=*/10);
  EXPECT_EQ(spline.Size(), 0);
  EXPECT_EQ(spline.Dim(), 10);
  std::vector<double> values(10, 1.0);
  spline.Sample(2.0, absl::MakeSpan(values));
  for (double v : values) {
    EXPECT_EQ(v, 0.0);
  }
}

TEST_P(TimeSplineAllInterpolationsTest, OneNode) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.SetInterpolation(test_case.interpolation);
  EXPECT_EQ(spline.Interpolation(), test_case.interpolation);

  spline.AddNode(1.0, {1.0, 2.0});
  EXPECT_EQ(spline.Size(), 1);
  for (double time : {0.0, 2.0, 4.0}) {
    EXPECT_THAT(spline.Sample(time), ElementsAre(1.0, 2.0));
  }
}

TEST_P(TimeSplineAllInterpolationsTest, TwoNodes) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.SetInterpolation(test_case.interpolation);

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
  spline.SetInterpolation(SplineInterpolation::kZeroSpline);
  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {3.0, 4.0});

  EXPECT_THAT(spline.Sample(1.5), ElementsAre(1.0, 2.0));
}

TEST(TimeSplineTest, Linear) {
  TimeSpline spline(/*dim=*/2);
  spline.SetInterpolation(SplineInterpolation::kLinearSpline);
  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {3.0, 4.0});

  EXPECT_THAT(spline.Sample(1.5), ElementsAre(2.0, 3.0));
}

TEST(TimeSplineTest, Cubic) {
  TimeSpline spline(/*dim=*/2);
  spline.SetInterpolation(SplineInterpolation::kCubicSpline);
  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {3.0, 4.0});

  EXPECT_THAT(spline.Sample(1.5), ElementsAre(2.0, 3.0));

  spline.Clear();
  spline.AddNode(0.0, {1.0, 2.0});
  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {3.0, 4.0});
  spline.AddNode(3.0, {3.0, 4.0});

  EXPECT_THAT(spline.Sample(1.5), ElementsAre(2.0, 3.0));

  spline = TimeSpline(/*dim=*/1);
  spline.SetInterpolation(SplineInterpolation::kCubicSpline);
  spline.AddNode(-1.0, {1.0});
  spline.AddNode(0.0, {0.0});
  spline.AddNode(1.0, {1.0});
  for (double x = 0.0; x <= 1.0; x += 0.125) {
    // Known solution for this spline
    double y = -std::pow(x, 3) + 2 * std::pow(x, 2);
    EXPECT_THAT(spline.Sample(x), ElementsAre(y));
  }
}

TEST(TimeSplineAllInterpolationsTest, ShiftTime) {
  TimeSpline spline(/*dim=*/2);
  spline.SetInterpolation(mjpc::spline::kLinearSpline);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});
  spline.AddNode(3.0, {3.0, 4.0});
  spline.AddNode(4.0, {4.0, 5.0});
  EXPECT_EQ(spline.Size(), 4);

  EXPECT_THAT(spline.Sample(1.0), ElementsAre(1.0, 2.0));
  EXPECT_THAT(spline.Sample(1.5), ElementsAre(1.5, 2.5));

  // Shift the spline so that the first node is at 1.5.
  spline.ShiftTime(1.5);
  EXPECT_EQ(spline.Size(), 4);
  EXPECT_THAT(spline.Sample(1.5), ElementsAre(1.0, 2.0));
  EXPECT_THAT(spline.Sample(2.0), ElementsAre(1.5, 2.5));
}

TEST_P(TimeSplineAllInterpolationsTest, DiscardBefore) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.SetInterpolation(test_case.interpolation);

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
  if (test_case.interpolation == SplineInterpolation::kCubicSpline) {
    // Cubic spline should keep one extra value before the given time passed
    // to DiscardBefore.
    EXPECT_EQ(discarded, 1);
    EXPECT_EQ(spline.Size(), 3);
    EXPECT_THAT(spline.Sample(1.0), ElementsAre(2.0, 3.0));
  } else {
    EXPECT_EQ(discarded, 2);
    EXPECT_EQ(spline.Size(), 2);
    EXPECT_THAT(spline.Sample(1.0), ElementsAre(3.0, 4.0));
  }

  // Discarding just after 3.0 should have no effect.
  EXPECT_EQ(spline.DiscardBefore(3.9), 0);
  if (test_case.interpolation == SplineInterpolation::kCubicSpline) {
    // Cubic spline should keep one extra value before the given time passed
    // to DiscardBefore.
    EXPECT_EQ(spline.Size(), 3);
    EXPECT_THAT(spline.Sample(1.0), ElementsAre(2.0, 3.0));
  } else {
    EXPECT_EQ(spline.Size(), 2);
    EXPECT_THAT(spline.Sample(1.0), ElementsAre(3.0, 4.0));
  }
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
  spline.SetInterpolation(SplineInterpolation::kLinearSpline);
  spline.Reserve(3);
  EXPECT_EQ(spline.Size(), 0);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});

  spline.Reserve(6);
  spline.AddNode(3.0, {4.0, 5.0});
  EXPECT_EQ(spline.Size(), 3);

  EXPECT_THAT(spline.Sample(2.5), ElementsAre(3.0, 4.0));
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
  spline.SetInterpolation(SplineInterpolation::kLinearSpline);
  spline.Reserve(test_case.reserve);

  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {2.0, 3.0});
  spline.DiscardBefore(2.0);
  spline.AddNode(3.0, {3.0, 4.0});
  spline.AddNode(4.0, {4.0, 5.0});
  spline.AddNode(5.0, {5.0, 6.0});

  TimeSpline spline2(spline);
  EXPECT_EQ(spline2.Size(), 4);

  // overwrite values in original spline
  for (TimeSpline::Node& n : spline) {
    n.values()[0] = 3.0;
    n.values()[1] = 4.0;
  }

  // spline2 should be unaffected
  EXPECT_THAT(spline2.Sample(1.5), ElementsAre(2.0, 3.0));
  EXPECT_THAT(spline2.Sample(2.5), ElementsAre(2.5, 3.5));
}

TEST_P(TimeSplineReserveTest, CopyAssignment) {
  const TimeSplineTestCase& test_case = GetParam();
  TimeSpline spline(/*dim=*/2);
  spline.SetInterpolation(SplineInterpolation::kLinearSpline);
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

  // overwrite values in original spline
  spline.Clear();

  // spline2 should be unaffected
  EXPECT_THAT(spline2.Sample(1.5), ElementsAre(2.0, 3.0));
  EXPECT_THAT(spline2.Sample(2.5), ElementsAre(2.5, 3.5));
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
  spline.SetInterpolation(SplineInterpolation::kZeroSpline);
  spline.Reserve(10);

  spline.AddNode(1.0, {});
  spline.AddNode(2.0, {});
  EXPECT_EQ(spline.Size(), 2);
  EXPECT_EQ(spline.DiscardBefore(2.0), 1);
  EXPECT_EQ(spline.Size(), 1);
  std::vector<double> values;
  spline.Sample(1, absl::MakeSpan(values.data(), values.size()));
}

template <typename T>
T BeginIterator(TimeSpline& spline) {
  if constexpr (std::is_same_v<T, TimeSpline::const_iterator>) {
    return spline.cbegin();
  } else {
    return spline.begin();
  }
}

template <typename T>
T EndIterator(TimeSpline& spline) {
  if constexpr (std::is_same_v<T, TimeSpline::const_iterator>) {
    return spline.cend();
  } else {
    return spline.end();
  }
}

template <typename T>
void TestIterator() {
  // Tests that the iterator type T complies with random_access_iterator_tag.
  TimeSpline spline(/*dim=*/2);
  spline.Reserve(10);
  spline.AddNode(1.0, {1.0, 2.0});
  spline.AddNode(2.0, {3.0, 4.0});
  spline.AddNode(3.0, {5.0, 6.0});

  T it = BeginIterator<T>(spline);
  EXPECT_EQ(it->values()[0], 1.0);
  EXPECT_EQ((it + 2)->values()[0], 5.0);
  EXPECT_EQ((2 + it)->values()[0], 5.0);
  EXPECT_EQ(EndIterator<T>(spline) - it, 3);
  EXPECT_EQ((EndIterator<T>(spline) - 1) - it, 2);

  it++;
  EXPECT_EQ(it->values()[0], 3.0);
  ++it;
  EXPECT_EQ(it->values()[0], 5.0);
  it--;
  EXPECT_EQ(it->values()[0], 3.0);
  --it;
  EXPECT_EQ(it->values()[0], 1.0);

  EXPECT_LT(it, it + 2);
  EXPECT_LE(it, it + 2);
  EXPECT_LE(it, it + 0);
  EXPECT_GT(it + 2, it);
  EXPECT_GE(it + 2, it);
  EXPECT_GE(it + 0, it);
  EXPECT_EQ((it + 2) - 2, it);

  TimeSpline spline2 = spline;
  EXPECT_NE(BeginIterator<T>(spline), BeginIterator<T>(spline2))
      << "Iterators from different splines should not be equal";

  // Copy constructor
  T it_copy = it;
  EXPECT_EQ(it, it_copy);

  auto node2 = it[2];
  EXPECT_EQ(node2.values()[0], 5.0);

  // Iterators are swappable
  it_copy += spline.Size();
  std::swap(it_copy, it);
  EXPECT_EQ(it_copy, BeginIterator<T>(spline));
  EXPECT_EQ(it, EndIterator<T>(spline));
}

TEST(TimeSplineTest, Iterator) {
  TestIterator<TimeSpline::iterator>();
}

TEST(TimeSplineTest, ConstIterator) {
  TestIterator<TimeSpline::const_iterator>();
}

INSTANTIATE_TEST_SUITE_P(
    TimeSplineAllInterpolations, TimeSplineAllInterpolationsTest,
    testing::ValuesIn<TimeSplineTestCase>({
        {"ZeroSpline", SplineInterpolation::kZeroSpline},
        {"LinearSpline", SplineInterpolation::kLinearSpline},
        {"CubicSpline", SplineInterpolation::kCubicSpline},
    }),
    [](const testing::TestParamInfo<TimeSplineAllInterpolationsTest::ParamType>&
           info) { return info.param.test_name; });

INSTANTIATE_TEST_SUITE_P(
    TimeSplineReserve, TimeSplineReserveTest,
    testing::ValuesIn<TimeSplineTestCase>({
        {"Reserve0", SplineInterpolation::kZeroSpline, 0},
        {"Reserve4", SplineInterpolation::kZeroSpline, 4},
        {"Reserve7", SplineInterpolation::kZeroSpline, 7},
    }),
    [](const testing::TestParamInfo<TimeSplineReserveTest::ParamType>&
           info) { return info.param.test_name; });

}  // namespace
