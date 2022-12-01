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

#ifndef MJPC_PLANNERS_ILQG_BOXQP_H_
#define MJPC_PLANNERS_ILQG_BOXQP_H_

#include <cstdlib>
#include <vector>

#include <mujoco/mujoco.h>

namespace mjpc {

// ----- boxQP data ----- //
// min 0.5 res' H res + res' g
//  st lower <= res <= upper
class BoxQP {
 public:
  // constructor
  BoxQP() = default;

  // destructor
  ~BoxQP() = default;

  // allocate memory
  void Allocate(int n) {
    // size memory
    res.resize(n);
    R.resize(n * (n + 7));
    H.resize(n * n);
    g.resize(n);
    lower.resize(n);
    upper.resize(n);
    index.resize(n);

    // reset for warmstart
    mju_zero(res.data(), n);
  }

  // ----- members ----- //
  std::vector<double> res;    // solution
  std::vector<double> R;      // factorization
  std::vector<int> index;     // free indices
  std::vector<double> H;      // SPD matrix
  std::vector<double> g;      // bias
  std::vector<double> lower;  // lower bounds
  std::vector<double> upper;  // upper bounds
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_ILQG_BOXQP_H_
