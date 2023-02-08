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

#include "mjpc/norm.h"

#include <algorithm>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "mjpc/test/finite_difference.h"

namespace {

struct NormTestCase {
  std::string test_name;
  mjpc::NormType norm_type;
  double params[2];
};

using NormTest = ::testing::TestWithParam<NormTestCase>;

bool AbsCompare(double a, double b) {
  return std::abs(a) < std::abs(b);
}

constexpr int kNPoints = 5;
constexpr int kDims = 2;
constexpr double kPoints[kNPoints][kDims] = {
  {0, 0}, {1, 0}, {-1, 0}, {1, 1}, {-1, -1}};

TEST_P(NormTest, Gradient) {
  // Hessian
  mjpc::FiniteDifferenceGradient fd;

  const double* params = GetParam().params;
  const mjpc::NormType norm_type = GetParam().norm_type;
  auto eval = [&](const double* x, int n) {
    return mjpc::Norm(nullptr, nullptr, x, params, kDims, norm_type);
  };
  // allocate
  fd.Allocate(eval, 2, 1.0e-6);

  // evaluate
  for (int i = 0; i < kNPoints; ++i) {
    const double *x = kPoints[i];
    fd.Gradient(x);
    std::vector<double> g(kDims);
    mjpc::Norm(g.data(), nullptr, x, params, kDims, norm_type);

    double abs_max =
        std::abs(*std::max_element(g.begin(), g.end(), AbsCompare));
    EXPECT_THAT(g, testing::Pointwise(testing::DoubleNear(abs_max * 1e-3),
                                      fd.gradient));
  }
}

TEST_P(NormTest, Hessian) {
  // Hessian
  mjpc::FiniteDifferenceHessian fd;

  const double* params = GetParam().params;
  double g[kDims] = {0};
  const mjpc::NormType norm_type = GetParam().norm_type;
  auto eval = [&](const double* x, int n) {
    return mjpc::Norm(nullptr, nullptr, x, params, kDims, norm_type);
  };
  // allocate
  fd.Allocate(eval, kDims, 1.0e-4);

  // evaluate
  for (int i = 0; i < kNPoints; ++i) {
    const double *x = kPoints[i];
    fd.Hessian(x);
    std::vector<double> H;
    H.resize(kDims * kDims);
    mjpc::Norm(g, H.data(), x, params, kDims, norm_type);

    // test Hessian
    double abs_max =
        std::abs(*std::max_element(H.begin(), H.end(), AbsCompare));
    EXPECT_THAT(
        H, testing::Pointwise(testing::DoubleNear(abs_max * 1e-2), fd.hessian));
  }
}

INSTANTIATE_TEST_SUITE_P(
    NormTest, NormTest,
    testing::ValuesIn<NormTestCase>({
        {"QUADRATIC_NORM", mjpc::NormType::kQuadratic, {0.1}},
        {"L22_NORM", mjpc::NormType::kL22, {0.1, 2}},
        {"L2_NORM", mjpc::NormType::kL2, {0.1}},
        {"COSH_NORM", mjpc::NormType::kCosh, {0.1}},
        {"POWER_LOSS", mjpc::NormType::kPowerLoss, {2}},
        {"SMOOTH_ABS_LOSS", mjpc::NormType::kSmoothAbsLoss, {0.1}},
        {"SMOOTH_ABS2_LOSS", mjpc::NormType::kSmoothAbs2Loss, {0.1, 2}},
        {"RECTIFY_LOSS", mjpc::NormType::kRectifyLoss, {0.1}},
        {"RATIO_LOSS", mjpc::NormType::kRatioLoss, {0.1}},
    }),
    [](const testing::TestParamInfo<NormTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
