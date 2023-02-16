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

#include "mjpc/planners/gradient/spline_mapping.h"

#include <algorithm>
#include <vector>

#include "mjpc/trajectory.h"

namespace mjpc {

// allocate memory
void ZeroSplineMapping::Allocate(int dim) {
  // dimensions
  this->dim = dim;

  // allocate
  mapping.resize((dim * kMaxTrajectoryHorizon) *
                 (dim * kMaxGradientSplinePoints));
}

// compute zero-order-hold mapping
void ZeroSplineMapping::Compute(const std::vector<double>& input_times,
                                int num_input, const double* output_times,
                                int num_output) {
  // set zeros
  std::fill(mapping.begin(),
            mapping.begin() + (dim * num_output) * (dim * num_input), 0.0);

  // compute
  int row, col;
  int bounds[2];
  for (int i = 0; i < num_output; i++) {
    FindInterval(bounds, input_times, output_times[i], num_input);
    for (int j = 0; j < dim; j++) {
      // p0
      row = dim * num_input * (dim * i + j);
      col = dim * bounds[0] + j;
      mapping[row + col] = 1.0;
    }
  }
}

// allocate memory
void LinearSplineMapping::Allocate(int dim) {
  // dimensions
  this->dim = dim;

  // allocate
  mapping.resize((dim * kMaxTrajectoryHorizon) *
                 (dim * kMaxGradientSplinePoints));
}

// compute linear-interpolation mapping
void LinearSplineMapping::Compute(const std::vector<double>& input_times,
                                  int num_input, const double* output_times,
                                  int num_output) {
  // set zeros
  std::fill(mapping.begin(),
            mapping.begin() + (dim * num_output) * (dim * num_input), 0.0);
  // compute
  int row, col;
  int bounds[2];
  for (int i = 0; i < num_output; i++) {
    FindInterval(bounds, input_times, output_times[i], num_input);
    for (int j = 0; j < dim; j++) {
      if (bounds[0] == bounds[1]) {
        // p1
        row = dim * num_input * (dim * i + j);
        col = dim * bounds[0] + j;
        mapping[row + col] = 1.0;
      } else {
        // normalized time
        double a = (output_times[i] - input_times[bounds[0]]) /
                   (input_times[bounds[1]] - input_times[bounds[0]]);

        // p0
        row = dim * num_input * (dim * i + j);
        col = dim * bounds[0] + j;
        mapping[row + col] = 1.0 - a;

        // p1
        row = dim * num_input * (dim * i + j);
        col = dim * bounds[1] + j;
        mapping[row + col] = a;
      }
    }
  }
}

// allocate memory
void CubicSplineMapping::Allocate(int dim) {
  // dimensions
  this->dim = dim;

  // allocate
  mapping.resize((dim * kMaxTrajectoryHorizon) *
                 (dim * kMaxGradientSplinePoints));
  point_slope_mapping.resize((2 * dim * kMaxGradientSplinePoints) *
                             (dim * kMaxGradientSplinePoints));
  output_mapping.resize((dim * kMaxTrajectoryHorizon) *
                        (2 * dim * kMaxGradientSplinePoints));
}

// compute cubic-interpolation mapping
void CubicSplineMapping::Compute(const std::vector<double>& input_times,
                                 int num_input, const double* output_times,
                                 int num_output) {
  // FiniteDifferenceSlope matrix
  std::fill(
      point_slope_mapping.begin(),
      point_slope_mapping.begin() + (2 * dim * num_input) * (dim * num_input),
      0.0);
  int row, col;
  // point-to-point mapping
  for (int i = 0; i < num_input; i++) {
    for (int j = 0; j < dim; j++) {
      row = dim * num_input * (dim * i + j);
      col = dim * i + j;
      point_slope_mapping[row + col] = 1.0;
    }
  }

  // point-to-FiniteDifferenceSlope mapping
  int shift = (dim * num_input) * (dim * num_input);
  for (int i = 0; i < num_input; i++) {
    double dt1 = (i > 0 ? 1.0 / (input_times[i] - input_times[i - 1]) : 0.0);
    double dt2 =
        (i < num_input - 1 ? 1.0 / (input_times[i + 1] - input_times[i]) : 0.0);
    if (i > 0 && i < num_input - 1) {
      dt1 *= 0.5;
      dt2 *= 0.5;
    }
    for (int j = 0; j < dim; j++) {
      // i - 1
      if (i - 1 >= 0) {
        row = dim * num_input * (dim * i + j);
        col = dim * (i - 1) + j;
        point_slope_mapping[shift + row + col] = -dt1;
      }

      // i
      row = dim * num_input * (dim * i + j);
      col = dim * i + j;
      point_slope_mapping[shift + row + col] = dt1 - dt2;

      // i + 1
      if (i + 1 <= num_input - 1) {
        row = dim * num_input * (dim * i + j);
        col = dim * (i + 1) + j;
        point_slope_mapping[shift + row + col] = dt2;
      }
    }
  }

  // output matrix
  std::fill(output_mapping.begin(),
            output_mapping.begin() + (dim * num_output) * (2 * dim * num_input),
            0.0);
  int bounds[2];
  double coefficients[4];
  for (int i = 0; i < num_output; i++) {
    FindInterval(bounds, input_times, output_times[i], num_input);
    CubicCoefficients(coefficients, output_times[i], input_times, num_input);
    for (int j = 0; j < dim; j++) {
      // p0
      row = 2 * dim * num_input * (dim * i + j);
      col = dim * bounds[0] + j;
      output_mapping[row + col] = coefficients[0];

      // m0
      row = 2 * dim * num_input * (dim * i + j);
      col = dim * num_input + dim * bounds[0] + j;
      output_mapping[row + col] = coefficients[1];

      if (bounds[0] != bounds[1]) {
        // p1
        row = 2 * dim * num_input * (dim * i + j);
        col = dim * bounds[1] + j;
        output_mapping[row + col] = coefficients[2];

        // m1
        row = 2 * dim * num_input * (dim * i + j);
        col = dim * num_input + dim * bounds[1] + j;
        output_mapping[row + col] = coefficients[3];
      }
    }
  }

  // mapping
  mju_mulMatMat(mapping.data(), output_mapping.data(),
                point_slope_mapping.data(), dim * num_output,
                2 * dim * num_input, dim * num_input);
}

}  // namespace mjpc
