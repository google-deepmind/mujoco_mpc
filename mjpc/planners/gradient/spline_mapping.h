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

#ifndef MJPC_PLANNERS_GRADIENT_SPLINE_MAPPING_H_
#define MJPC_PLANNERS_GRADIENT_SPLINE_MAPPING_H_

#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/utilities.h"

namespace mjpc {

// ----- spline constants ----- //
inline constexpr int kMinGradientSplinePoints = 1;
inline constexpr int kMaxGradientSplinePoints = 25;
inline constexpr int kMinGradientSplinePower = 1;
inline constexpr int kMaxGradientSplinePower = 5;

// matrix representation for mapping between spline points and interpolated time
// series
class SplineMapping {
 public:
  // constructor
  SplineMapping() {}

  // destructor
  virtual ~SplineMapping() {}

  // ----- methods ----- //

  // allocate memory
  virtual void Allocate(int dim) = 0;

  // compute mapping
  virtual void Compute(const std::vector<double>& input_times, int num_input,
                       const double* output_times, int num_output) = 0;

  // return mapping
  virtual double* Get() = 0;
};

// zero-order-hold mapping
class ZeroSplineMapping : public SplineMapping {
 public:
  // constructor
  ZeroSplineMapping() {}

  // destructor
  ~ZeroSplineMapping() {}

  // ----- methods ----- //
  // allocate memory
  void Allocate(int dim);

  // compute mapping
  void Compute(const std::vector<double>& input_times, int num_input,
               const double* output_times, int num_output);

  // return mapping
  double* Get() { return mapping.data(); }

  // ----- members ----- //
  std::vector<double> mapping;
  int dim;
};

// linear-interpolation mapping
class LinearSplineMapping : public SplineMapping {
 public:
  // constructor
  LinearSplineMapping() {}

  // destructor
  ~LinearSplineMapping() {}

  // ----- methods ----- //

  // allocate memory
  void Allocate(int dim);

  // compute mapping
  void Compute(const std::vector<double>& input_times, int num_input,
               const double* output_times, int num_output);

  // return mapping
  double* Get() { return mapping.data(); }

  // ----- members ----- //
  std::vector<double> mapping;
  int dim;
};

// cubic-interpolation mapping
class CubicSplineMapping : public SplineMapping {
 public:
  // constructor
  CubicSplineMapping() {}

  // destructor
  ~CubicSplineMapping() {}

  // ----- methods ----- //

  // allocate memory
  void Allocate(int dim);

  // compute mapping
  void Compute(const std::vector<double>& input_times, int num_input,
               const double* output_times, int num_output);

  // return mapping
  double* Get() { return mapping.data(); }

  // ----- members ----- //
  std::vector<double> mapping;
  std::vector<double> point_slope_mapping;
  std::vector<double> output_mapping;
  int dim;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_GRADIENT_SPLINE_MAPPING_H_
