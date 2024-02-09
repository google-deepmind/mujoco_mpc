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

#ifndef MJPC_PLANNERS_PLANNER_H_
#define MJPC_PLANNERS_PLANNER_H_

#include <mujoco/mujoco.h>

#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

inline constexpr int kMaxTrajectory = 128;
inline constexpr int kMaxTrajectoryLarge = 1028;

// virtual planner
class Planner {
 public:
  // destructor
  virtual ~Planner() = default;

  // initialize data and settings
  virtual void Initialize(mjModel* model, const Task& task) = 0;

  // allocate memory
  virtual void Allocate() = 0;

  // reset memory to zeros
  virtual void Reset(int horizon,
                     const double* initial_repeated_action = nullptr) = 0;

  // set state
  virtual void SetState(const State& state) = 0;

  // optimize nominal policy
  virtual void OptimizePolicy(int horizon, ThreadPool& pool) = 0;

  // compute trajectory using nominal policy
  virtual void NominalTrajectory(int horizon, ThreadPool& pool) = 0;

  // set action from policy
  virtual void ActionFromPolicy(double* action, const double* state,
                                double time, bool use_previous = false) = 0;

  // return trajectory with best total return, or nullptr if no planning
  // iteration has completed
  virtual const Trajectory* BestTrajectory() = 0;

  // visualize planner-specific traces
  virtual void Traces(mjvScene* scn) = 0;

  // planner-specific GUI elements
  virtual void GUI(mjUI& ui) = 0;

  // planner-specific plots
  virtual void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                     int planner_shift, int timer_shift, int planning,
                     int* shift) = 0;

  // return number of parameters optimized by planner
  virtual int NumParameters() = 0;

  std::vector<UniqueMjData> data_;
  void ResizeMjData(const mjModel* model, int num_threads);
};

// additional optional interface for planners that can produce several policy
// proposals
class RankedPlanner : public Planner {
 public:
  virtual ~RankedPlanner() = default;
  // optimizes policies, but rather than picking the best, generate up to
  // ncandidates. returns number of candidates created. only called
  // from the planning thread.
  virtual int OptimizePolicyCandidates(int ncandidates, int horizon,
                                        ThreadPool& pool) = 0;
  // returns the total return for the nth candidate (or another score to
  // minimize). only called from the planning thread.
  virtual double CandidateScore(int candidate) const = 0;

  // set action from candidate policy. only called from the planning thread.
  virtual void ActionFromCandidatePolicy(double* action, int candidate,
                                         const double* state, double time) = 0;

  // sets the nth candidate to the active policy.
  virtual void CopyCandidateToPolicy(int candidate) = 0;
};

}  // namespace mjpc

#endif  // MJPC_PLANNERS_PLANNER_H_
