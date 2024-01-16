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

#include "mjpc/planners/robust/robust_planner.h"
#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/planners/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

namespace mju = ::mujoco::util_mjpc;

void RobustPlanner::Initialize(mjModel* model, const Task& task) {
  delegate_->Initialize(model, task);
  // delete mjData instances since model might have changed.
  data_.clear();
  // allocate one mjData for nominal.
  ResizeMjData(model, 1);

  trajectories_.clear();

  // model
  model_ = model;

  // task
  task_ = &task;

  nrepetitions_ = GetNumberOrDefault(5, model, "robust_repetitions");
  // if robust_candidates is not defined, derive it from number of rollouts
  // in sampling config
  ncandidates_ = GetNumberOrDefault(-1, model, "robust_candidates");
  if (ncandidates_ == -1) {
    int sampling_rollouts =
        GetNumberOrDefault(10, model, "sampling_trajectories");
    ncandidates_ = sampling_rollouts / nrepetitions_;
  }

  xfrc_std_ = GetNumberOrDefault(0.1, model, "robust_xfrc");
  xfrc_rate_ = GetNumberOrDefault(0.1, model, "robust_xfrc_rate");
}

void RobustPlanner::Allocate() {
  delegate_->Allocate();
  // initial state
  int num_state = model_->nq + model_->nv + model_->na;

  // state
  state_.resize(num_state);
  mocap_.resize(7 * model_->nmocap);
  userdata_.resize(model_->nuserdata);

  ResizeTrajectories(ncandidates_ * nrepetitions_);
}

void RobustPlanner::Reset(int horizon, const double* initial_repeated_action) {
  delegate_->Reset(horizon, initial_repeated_action);
  // state
  std::fill(state_.begin(), state_.end(), 0.0);
  std::fill(mocap_.begin(), mocap_.end(), 0.0);
  std::fill(userdata_.begin(), userdata_.end(), 0.0);
  time_ = 0.0;

  for (auto& trajectory : trajectories_) {
    trajectory.Reset(kMaxTrajectoryHorizon);
  }
}

void RobustPlanner::SetState(const State& state) {
  delegate_->SetState(state);
  state.CopyTo(state_.data(), mocap_.data(), userdata_.data(), &time_);
}

void RobustPlanner::OptimizePolicy(int horizon, ThreadPool& pool) {
  // get the best N candidates
  int ncandidates =
      delegate_->OptimizePolicyCandidates(ncandidates_, horizon, pool);
  if (!ncandidates) {
    return;
  }
  // if there's only one candidate, use that one.
  if (ncandidates == 1) {
    delegate_->CopyCandidateToPolicy(0);
    return;
  }

  // For each candidate, roll out several trajectories with force perturbations
  // TODO(nimrod): Add domain randomization to the model for these rollouts
  ResizeMjData(model_, pool.NumThreads());

  int repetitions = nrepetitions_;
  ResizeTrajectories(ncandidates * repetitions);

  int count_before = pool.GetCount();
  for (int i = 0; i < ncandidates; i++) {
    for (int j = 0; j < repetitions; j++) {
      Trajectory* trajectory = &trajectories_[repetitions*i + j];
      pool.Schedule([&, delegate = delegate_.get(), candidate = i,
                     trajectory]() {
        auto sample_policy_i = [delegate, candidate](double* action,
                                                     const double* state,
                                                     double time) {
          delegate->ActionFromCandidatePolicy(action, candidate, state, time);
        };
        trajectory->NoisyRollout(
            sample_policy_i, task_, model_, data_[ThreadPool::WorkerId()].get(),
            state_.data(), time_, mocap_.data(), userdata_.data(),
            /*xfrc_std=*/xfrc_std_, /*xfrc_rate=*/xfrc_rate_, horizon);
      });
    }
  }
  pool.WaitCount(count_before + ncandidates * repetitions);
  pool.ResetCount();

  // for each candidate find the worst performing rollout. pick the
  // candidate with the best worst performing rollout.
  int best_candidate = -1;
  double best_score = 0;
  for (int candidate = 0; candidate < ncandidates; candidate++) {
    double mean_return = delegate_->CandidateScore(candidate);
    int valid_rollouts = 0;
    for (int j = 0; j < repetitions; j++) {
      // if a rollout fails, don't affect the candidate's score
      if (trajectories_[repetitions * candidate + j].failure) {
        continue;
      }
      double total_return =
          trajectories_[repetitions * candidate + j].total_return;
      mean_return =
          (valid_rollouts * mean_return + total_return) / (valid_rollouts + 1);
      valid_rollouts++;
    }
    if (best_candidate == -1 || mean_return < best_score) {
      best_candidate = candidate;
      best_score = mean_return;
    }
  }

  delegate_->CopyCandidateToPolicy(best_candidate);
}

void RobustPlanner::NominalTrajectory(int horizon, ThreadPool& pool) {
  delegate_->NominalTrajectory(horizon, pool);
}
void RobustPlanner::ActionFromPolicy(double* action, const double* state,
                                     double time, bool use_previous) {
  delegate_->ActionFromPolicy(action, state, time, use_previous);
}
const Trajectory* RobustPlanner::BestTrajectory() {
  return delegate_->BestTrajectory();
}
void RobustPlanner::Traces(mjvScene* scn) { delegate_->Traces(scn); }
void RobustPlanner::GUI(mjUI& ui) {
  delegate_->GUI(ui);
  mjuiDef defRobust[] = {
      {mjITEM_SLIDERINT, "R Candidates", 2, &ncandidates_, "0 1"},
      {mjITEM_SLIDERINT, "R Rollouts", 2, &nrepetitions_, "1 10"},
      {mjITEM_SLIDERNUM, "R XFRC Std", 2, &xfrc_std_, "0 1"},
      {mjITEM_SLIDERNUM, "R XFRC Rate", 2, &xfrc_rate_, "0 1"},
      {mjITEM_END}};

  // set number of candidates slider limits
  mju::sprintf_arr(defRobust[0].other, "%i %i", 1, kMaxTrajectory);

  // add robust planner
  mjui_add(&ui, defRobust);
}
void RobustPlanner::Plots(mjvFigure* fig_planner, mjvFigure* fig_timer,
                          int planner_shift, int timer_shift, int planning,
                          int* shift) {
  delegate_->Plots(fig_planner, fig_timer, planner_shift, timer_shift, planning,
                   shift);
}

void RobustPlanner::ResizeTrajectories(int ntrajectories) {
  int size_before = trajectories_.size();
  if (size_before < ntrajectories) {
    trajectories_.resize(ntrajectories);
    int num_state = model_->nq + model_->nv + model_->na;
    for (int i = size_before; i < ntrajectories; i++) {
      Trajectory& trajectory = trajectories_[i];
      trajectory.Initialize(num_state, model_->nu, task_->num_residual,
                              task_->num_trace, kMaxTrajectoryHorizon);
      trajectory.Allocate(kMaxTrajectoryHorizon);
    }
  }
}

}  // namespace mjpc
