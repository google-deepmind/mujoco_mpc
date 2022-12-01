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

#ifndef MJPC_TEST_FINITE_DIFFERENCE_H_
#define MJPC_TEST_FINITE_DIFFERENCE_H_

#include <cstdlib>
#include <functional>
#include <vector>

namespace mjpc {

// finite difference gradient for scalar output functions
class FiniteDifferenceGradient {
 public:
  // contstructor
  FiniteDifferenceGradient() = default;

  // destructor
  ~FiniteDifferenceGradient() = default;

  // ----- methods ----- //

  // allocate memory, set function and settings
  void Allocate(std::function<double(const double*, int)> f, int n, double eps);

  // compute gradient
  void Gradient(const double* x);

  // ----- members ----- //
  std::function<double(const double*, int)> eval;
  std::vector<double> gradient;
  std::vector<double> workspace;
  int dimension;
  double epsilon;
};

// finite difference gradient for scalar output functions
class FiniteDifferenceHessian {
 public:
  // contstructor
  FiniteDifferenceHessian() = default;

  // destructor
  ~FiniteDifferenceHessian() = default;

  // ----- methods ----- //

  // allocate memory, set function and settings
  void Allocate(std::function<double(const double*, int)> f, int n, double eps);

  // compute gradient
  void Hessian(const double* x);

  // ----- members ----- //
  std::function<double(const double*, int)> eval;
  std::vector<double> hessian;
  std::vector<double> workspace1;
  std::vector<double> workspace2;
  std::vector<double> workspace3;

  int dimension;
  double epsilon;
};

}  // namespace mjpc

#endif  // MJPC_TEST_FINITE_DIFFERENCE_H_
