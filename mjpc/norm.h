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

#ifndef MJPC_NORM_H_
#define MJPC_NORM_H_

namespace mjpc {

// norm types
enum NormType : int {
  kNull = -1,
  kQuadratic = 0,
  kL22 = 1,
  kL2 = 2,
  kCosh = 3,
  kGeodesic = 4,
  kPowerLoss = 5,
  kSmoothAbsLoss = 6,
  kSmoothAbs2Loss = 7,
  kRectifyLoss = 8,
  kRatioLoss = 9,
};

// norm's number of parameters
int NormParameterDimension(int type);

// evaluate norm; optionally, gradient and Hessian
double Norm(double *g, double *H, const double *x, const double *params, int n,
            NormType type);

}  // namespace mjpc

#endif  // MJPC_NORM_H_
