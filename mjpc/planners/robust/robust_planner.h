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

// A Planner implementation that wraps another Planner, and reranks its policy
// proposals by running them with perturbations (and in the future - with
// domain-randomized models).
// This is a work in progress, and hasn't been shown to be useful yet.


#ifndef MJPC_MJPC_PLANNERS_ROBUST_ROBUST_PLANNER_H_
#define MJPC_MJPC_PLANNERS_ROBUST_ROBUST_PLANNER_H_

#include <memory>
#include <utility>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"

namespace mjpc {

class RobustPlanner : public Planner {
 public:
  RobustPlanner(std::unique_ptr<RankedPlanner> delegate)
      : delegate_(std::move(delegate)) {}
  ~RobustPlanner() override = default;

  void Initialize(mjModel* model, const Task& task) override;
  void Allocate() override;
  void Reset(int horizon,
             const double* initial_repeated_action = nullptr) override;
  void SetState(const State& state) override;
  void OptimizePolicy(int horizon, ThreadPool& pool) override;
  void NominalTrajectory(int horizon, ThreadPool& pool) override;
  void ActionFromPolicy(double* action, const double* state, double time,
                        bool use_previous = false) override;
  const Trajectory* BestTrajectory() override;
  void Traces(mjvScene* scn) override;
  void GUI(mjUI& ui) override;
  void Plots(mjvFigure* fig_planner, mjvFigure* fig_timer, int planner_shift,
             int timer_shift, int planning, int* shift) override;
  int NumParameters() override { return delegate_->NumParameters(); };

 private:
  void ResizeTrajectories(int ntrajectories);

  const mjModel* model_;
  const Task* task_;

  std::unique_ptr<RankedPlanner> delegate_;
  // number of candidate policies to evaluate with perturbations
  int ncandidates_ = 12;
  // number of trajectories per candidate to evaluate
  int nrepetitions_ = 5;
  // standard deviation of gaussian noise force perturbations
  double xfrc_std_ = 0.1;
  double xfrc_rate_ = 0.1;

  std::vector<Trajectory> trajectories_;

  // state
  std::vector<double> state_;
  double time_;
  std::vector<double> mocap_;
  std::vector<double> userdata_;
};

}  // namespace mjpc

#endif  // MJPC_MJPC_PLANNERS_ROBUST_ROBUST_PLANNER_H_
