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

#include <absl/random/random.h>
#include <mujoco/mujoco.h>

#include <functional>
#include <vector>

#include "gtest/gtest.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace {

// class FiniteDifferenceGradient {
//  public:
//   // constructor
//   FiniteDifferenceGradient(int dim) {
//     // allocate memory
//     gradient_.resize(dim);
//     workspace_.resize(dim);
//   }

//   // destructor
//   ~FiniteDifferenceGradient() {}

//   // compute gradient
//   double* Compute(std::function<double(double* x)> func, double* input,
//                   int dim) {
//     // resize
//     if (dim != gradient_.size()) gradient_.resize(dim);
//     if (dim != workspace_.size()) workspace_.resize(dim);

//     // ----- compute ----- //
//     // set workspace
//     mju_copy(workspace_.data(), input, dim);

//     // nominal evaluation
//     double f = func(input);

//     // finite difference
//     for (int i = 0; i < dim; i++) {
//       // positive perturbation
//       workspace_[i] += epsilon_;
//       double fp = func(workspace_.data());

//       // gradient
//       gradient_[i] = (fp - f) / epsilon_;

//       // reset
//       workspace_[i] = input[i];
//     }
//     return gradient_.data();
//   }

//   // members
//   std::vector<double> gradient_;
//   std::vector<double> workspace_;
//   double epsilon_ = 1.0e-6;
// };

// class FiniteDifferenceJacobian {
//  public:
//   // constructor
//   FiniteDifferenceJacobian(int num_output, int num_input) {
//     jacobian_.resize(num_output * num_input);
//     jacobian_transpose_.resize(num_input * num_output);
//     output_.resize(num_output);
//     output_nominal_.resize(num_output);
//     workspace_.resize(num_input);
//   }

//   // destructor
//   ~FiniteDifferenceJacobian() {}

//   // compute Jacobian
//   double* Compute(std::function<void(double* output, const double* input)>
//   func,
//                   double* input, int num_output, int num_input) {
//     // resize
//     if (jacobian_.size() != num_output * num_input)
//       jacobian_.resize(num_output * num_input);
//     if (jacobian_transpose_.size() != num_output * num_input)
//       jacobian_transpose_.resize(num_output * num_input);
//     if (output_.size() != num_output) output_.resize(num_output);
//     if (output_nominal_.size() != num_output)
//       output_nominal_.resize(num_output);
//     if (workspace_.size() != num_input) workspace_.resize(num_input);

//     // copy workspace
//     mju_copy(workspace_.data(), input, num_input);

//     // nominal evaluation
//     mju_zero(output_nominal_.data(), num_output);
//     func(output_nominal_.data(), workspace_.data());

//     for (int i = 0; i < num_input; i++) {
//       // perturb input
//       workspace_[i] += epsilon_;

//       // evaluate
//       mju_zero(output_.data(), num_output);
//       func(output_.data(), workspace_.data());

//       // Jacobian
//       double* JT = jacobian_transpose_.data() + i * num_output;
//       mju_sub(JT, output_.data(), output_nominal_.data(), num_output);
//       mju_scl(JT, JT, 1.0 / epsilon_, num_output);

//       // reset workspace
//       workspace_[i] = input[i];
//     }

//     // transpose
//     mju_transpose(jacobian_.data(), jacobian_transpose_.data(), num_output,
//                   num_input);

//     return jacobian_.data();
//   }

//   // members
//   std::vector<double> jacobian_;
//   std::vector<double> jacobian_transpose_;
//   std::vector<double> output_;
//   std::vector<double> output_nominal_;
//   std::vector<double> workspace_;
//   double epsilon_ = 1.0e-6;
// };

// class FiniteDifferenceHessian {
//  public:
//   // constructor
//   FiniteDifferenceHessian(int dim) {
//     hessian_.resize(dim * dim);
//     workspace1_.resize(dim);
//     workspace2_.resize(dim);
//     workspace3_.resize(dim);
//   }

//   // destructor
//   ~FiniteDifferenceHessian() {}

//   // compute
//   double* Compute(std::function<double(double* x)> func, double* input,
//                   int dim) {
//     // resize
//     if (dim * dim != hessian_.size()) hessian_.resize(dim * dim);
//     if (dim != workspace1_.size()) workspace1_.resize(dim);
//     if (dim != workspace2_.size()) workspace2_.resize(dim);
//     if (dim != workspace3_.size()) workspace3_.resize(dim);

//     // set workspace
//     mju_copy(workspace1_.data(), input, dim);
//     mju_copy(workspace2_.data(), input, dim);
//     mju_copy(workspace3_.data(), input, dim);

//     // evaluate at candidate
//     double f = func(input);

//     // centered finite difference
//     for (int i = 0; i < dim; i++) {
//       for (int j = 0; j < dim; j++) {
//         if (i > j) continue;  // skip bottom triangle
//         // workspace 1
//         workspace1_[i] += epsilon_;
//         workspace1_[j] += epsilon_;

//         double fij = func(workspace1_.data());

//         // workspace 2
//         workspace2_[i] += epsilon_;
//         double fi = func(workspace2_.data());

//         // workspace 3
//         workspace3_[j] += epsilon_;
//         double fj = func(workspace3_.data());

//         // Hessian value
//         double H = (fij - fi - fj + f) / (epsilon_ * epsilon_);
//         hessian_[i * dim + j] = H;
//         hessian_[j * dim + i] = H;

//         // reset workspace 1
//         workspace1_[i] = input[i];
//         workspace1_[j] = input[j];

//         // reset workspace 2
//         workspace2_[i] = input[i];

//         // reset workspace 3
//         workspace3_[j] = input[j];
//       }
//     }
//     return hessian_.data();
//   }

//   // members
//   std::vector<double> hessian_;
//   std::vector<double> workspace1_;
//   std::vector<double> workspace2_;
//   std::vector<double> workspace3_;
//   double epsilon_ = 1.0e-6;
// };

const double FD_TOLERANCE = 1.0e-3;

TEST(FiniteDifferenceTest, Quadratic) {
  // quadratic
  auto quadratic = [](double* x) { return 0.5 * (x[0] * x[0] + x[1] * x[1]); };
  const int n = 2;
  double input[n] = {1.0, 1.0};

  // gradient
  FiniteDifferenceGradient fdg(2);
  double* grad = fdg.Compute(quadratic, input, n);

  EXPECT_NEAR(grad[0], 1.0, FD_TOLERANCE);
  EXPECT_NEAR(grad[1], 1.0, FD_TOLERANCE);

  // Hessian
  FiniteDifferenceHessian fdh(2);
  double* hess = fdh.Compute(quadratic, input, n);

  // test
  EXPECT_NEAR(hess[0], 1.0, FD_TOLERANCE);
  EXPECT_NEAR(hess[1], 0.0, FD_TOLERANCE);
  EXPECT_NEAR(hess[2], 0.0, FD_TOLERANCE);
  EXPECT_NEAR(hess[3], 1.0, FD_TOLERANCE);
}

TEST(FiniteDifferenceTest, Jacobian) {
  // set up
  const int num_output = 2;
  const int num_input = 2;
  double A[num_output * num_input] = {1.0, 2.0, 3.0, 4.0};
  auto f = [&A](double* output, const double* input) {
    mju_mulMatVec(output, A, input, num_output, num_input);
  };
  double input[2] = {1.0, 1.0};

  // Jacobian
  FiniteDifferenceJacobian fdj(num_output, num_input);
  double* jac = fdj.Compute(f, input, num_output, num_input);

  // test
  EXPECT_NEAR(jac[0], A[0], FD_TOLERANCE);
  EXPECT_NEAR(jac[1], A[1], FD_TOLERANCE);
  EXPECT_NEAR(jac[2], A[2], FD_TOLERANCE);
  EXPECT_NEAR(jac[3], A[3], FD_TOLERANCE);
}

}  // namespace
}  // namespace mjpc
