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

#ifndef MJPC_PLANNERS_SAMPLING_PLANNER_H_
#define MJPC_PLANNERS_SAMPLING_PLANNER_H_

#include <mujoco/mujoco.h>

#include <atomic>
#include <cstdint>
#include <shared_mutex>
#include <vector>

#include "mjpc/planners/planner.h"
#include "mjpc/planners/sampling/policy.h"
#include "mjpc/spline/spline.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/trajectory.h"

namespace mjpc {

// sampling planner limits
inline constexpr int MinSamplingSplinePoints = 1;
inline constexpr int MaxSamplingSplinePoints = 36;
inline constexpr double MinNoiseStdDev = 0.0;
inline constexpr double MaxNoiseStdDev = 1.0;

class SamplingPlanner : public RankedPlanner {
 public:
  // constructor
  SamplingPlanner() = default;

  // destructor
  ~SamplingPlanner() override = default;

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
  void ActionFromPolicy(double* action, const double* state,
                        double time, bool use_previous = false) override;

  // resample nominal policy
  void UpdateNominalPolicy(int horizon);

  // add noise to nominal policy
  void AddNoiseToPolicy(double start_time, int i);

  // compute candidate trajectories
  void Rollouts(int num_trajectory, int horizon, ThreadPool& pool);

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

  // optimizes policies, but rather than picking the best, generate up to
  // ncandidates. returns number of candidates created.
  int OptimizePolicyCandidates(int ncandidates, int horizon,
                               ThreadPool& pool) override;
  // returns the total return for the nth candidate (or another score to
  // minimize)
  double CandidateScore(int candidate) const override;

  // set action from candidate policy
  void ActionFromCandidatePolicy(double* action, int candidate,
                                 const double* state, double time) override;

  void CopyCandidateToPolicy(int candidate) override;

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
  SamplingPolicy candidate_policy[kMaxTrajectory];
  SamplingPolicy previous_policy;

  // scratch
  mjpc::spline::TimeSpline plan_scratch;

  // trajectories
  Trajectory trajectory[kMaxTrajectory];

  // order of indices of rolled out trajectories, ordered by total return
  std::vector<int> trajectory_order;

  // ----- noise ----- //
  double noise_exploration[2] = {0};  // stds for sampling: N(0, exploration)
  std::vector<double> noise;
mjpc::spline::SplineInterpolation interpolation_ =
      mjpc::spline::SplineInterpolation::kZeroSpline;

  // best trajectory
  int winner;

  // improvement
  double improvement;

  // flags
  int processed_noise_status;

  // timing
  std::atomic<double> noise_compute_time;
  double rollouts_compute_time;
  double policy_update_compute_time;

  // If true, use sliding plans (no resampling)
  std::uint8_t sliding_plan_ = false;

  int num_trajectory_;
  mutable std::shared_mutex mtx_;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_SAMPLING_PLANNER_H_
