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

#ifndef MJPC_PLANNERS_SAMPLE_GRADIENT_PLANNER_H_
#define MJPC_PLANNERS_SAMPLE_GRADIENT_PLANNER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <vector>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"

namespace mjpc {

class SampleGradientPlanner : public Planner {
 public:
  // constructor
  SampleGradientPlanner() = default;

  // destructor
  ~SampleGradientPlanner() override = default;

  // ----- methods ----- //

  // initialize data and settings
  void Initialize(mjModel* model, const Task& task) override;

  // allocate memory
  void Allocate() override;

  // reset memory to zeros
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;

  // set state
  void SetState(const State& state) override;

  // optimize nominal policy using random sampling
  void OptimizePolicy(int horizon, ThreadPool& pool) override;

  // compute trajectory using nominal policy
  void NominalTrajectory(int horizon, ThreadPool& pool) override;

  // set action from policy
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;

  // resample nominal policy
  void ResamplePolicy(SamplingPolicy& policy, int horizon,
                      int num_spline_points);

  // add noise to nominal policy
  void AddNoiseToPolicy(int i);

  // rollout candidate policies
  void Rollouts(int num_trajectory, int num_gradient, int horizon,
                ThreadPool& pool);

  // compute candidate trajectories along approximate gradient direction
  void GradientCandidates(int num_trajectory, int num_gradient, int horizon,
                          ThreadPool& pool);

  // return trajectory with best total return
  const Trajectory* BestTrajectory() override;

  // visualize planner-specific traces
  void Traces(mjvScene* scn) override;

  // planner-specific GUI elements
  void GUI(mjUI& ui) override;

  // planner-specific plots
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;

  // return number of parameters optimized by planner
  int NumParameters() override {
    return policy.num_spline_points * model->nu;
  };

  // ----- members ----- //
  mjModel* model;
  const Task* task;

  // state
  std::vector<double> state;
  double time;
  std::vector<double> mocap;
  std::vector<double> userdata;

  // policy
  SamplingPolicy policy;  // (Guarded by mtx_)
  std::vector<SamplingPolicy> candidate_policy;
  SamplingPolicy resampled_policy;
  SamplingPolicy previous_policy;

  // scratch
  mjpc::spline::TimeSpline plan_scratch;

  // trajectories
  std::vector<Trajectory> trajectory;

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // zero-mean Gaussian noise standard deviation
  double noise_exploration;
  std::vector<double> noise;
  mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;

  // improvement
  double improvement;

  // winner index
  int winner = 0;

  // flags
  int processed_noise_status;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double gradient_candidates_compute_time;
  double policy_update_compute_time;

  int num_trajectory_;
  int num_gradient_;  // number of gradient candidates
  mutable std::shared_mutex mtx_;

  // approximate gradient
  std::vector<double> gradient;
  std::vector<double> gradient_previous;
  double gradient_filter_ = 1.0;

  // gradient step size
  std::vector<double> step_size_;
  double gradient_max_step_size = 2.0;
  double gradient_min_step_size = 1.0e-3;

  // return weight
  std::vector<double> return_weight_;

  // nominal index
  const int idx_nominal = 0;

  // ----- winner type ----- //
  enum WinnerType : int {
    kNominal = 0,
    kPerturb,
    kGradient,
  };

  int winner_type_ = 0;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_SAMPLE_GRADIENT_PLANNER_H_
