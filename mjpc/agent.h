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
#ifndef MJPC_AGENT_H_
#define MJPC_AGENT_H_

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "planners/include.h"
#include "states/include.h"
#include "states/state.h"
#include "task.h"
#include "threadpool.h"

namespace mjpc {

// figures
struct AgentPlots {
  mjvFigure action;
  mjvFigure cost;
  mjvFigure planner;
  mjvFigure timer;
};

class Agent {
 public:
  friend class AgentTest;

  // constructor
  Agent() : planners_(mjpc::LoadPlanners()), states_(mjpc::LoadStates()) {}

  // destructor
  ~Agent() {
    if (model_) mj_deleteModel(model_);
  }

  // ----- methods ----- //

  // initialize data, settings, planners, states
  void Initialize(mjModel* model, const std::string& task_names,
                  const char planner_str[], ResidualFunction* residual,
                  TransitionFunction* transition);

  // allocate memory
  void Allocate();

  // reset data, settings, planners, states
  void Reset();

  void PlanIteration(ThreadPool* pool);

  // call planner to update nominal policy
  void Plan(std::atomic<bool>& exitrequest, std::atomic<int>& uiloadrequest);

  // modify the scene, e.g. add trace visualization
  void ModifyScene(mjvScene* scn);

  // graphical user interface elements for agent and task
  void Gui(mjUI& ui);

  // task-based GUI event
  void TaskEvent(mjuiItem* it, mjData* data, std::atomic<int>& uiloadrequest,
                 int& run);

  // agent-based GUI event
  void AgentEvent(mjuiItem* it, mjData* data, std::atomic<int>& uiloadrequest,
                  int& run);

  // initialize plots
  void PlotInitialize();

  // reset plot data to zeros
  void PlotReset();

  // plot current information
  void Plots(const mjData* data, int shift);

  // render plots
  void PlotShow(mjrRect* rect, mjrContext* con);

  void SetState(const mjData* data);

  mjpc::Planner& ActivePlanner() { return *planners_[planner_]; }
  mjpc::State& ActiveState() { return *states_[state_]; }

  Task& task() { return task_; }
  int max_threads() const { return max_threads_;}

  // status flags, logically should be bool, but mjUI needs int pointers
  int plan_enabled;
  int action_enabled;
  int visualize_enabled;
  int allocate_enabled;
  int plot_enabled;

 private:
  // model
  mjModel* model_ = nullptr;

  // integrator
  int integrator_;

  // planning horizon (continuous time)
  double horizon_;

  // planning steps (number of discrete timesteps)
  int steps_;

  // time step
  double timestep_;

  // task
  Task task_;

  // planners
  std::vector<std::unique_ptr<mjpc::Planner>> planners_;
  int planner_;

  // states
  std::vector<std::unique_ptr<mjpc::State>> states_;
  int state_;

  // timing
  double agent_compute_time_;
  double rollout_compute_time_;

  // objective
  std::vector<double> residual_;
  double cost_;
  std::vector<double> terms_;

  // planning iterations counter
  std::atomic_int count_;

  // names
  char task_names_[1024];
  char planner_names_[1024];

  // plots
  AgentPlots plots_;

  // max threads for planning
  int max_threads_;
};

}  // namespace mjpc

#endif  // MJPC_AGENT_H_
