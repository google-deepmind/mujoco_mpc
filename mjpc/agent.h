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
#include <string_view>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/planners/include.h"
#include "mjpc/states/include.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"

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
  explicit Agent(const mjModel* model, std::shared_ptr<Task> task);

  // destructor
  ~Agent() {
    if (model_) mj_deleteModel(model_);  // we made a copy in Initialize
  }

  // ----- methods ----- //

  // initialize data, settings, planners, states
  void Initialize(const mjModel* model);

  // allocate memory
  void Allocate();

  // reset data, settings, planners, states
  void Reset();

  // single planner iteration
  void PlanIteration(ThreadPool* pool);

  // call planner to update nominal policy
  void Plan(std::atomic<bool>& exitrequest, std::atomic<int>& uiloadrequest);

  // modify the scene, e.g. add trace visualization
  void ModifyScene(mjvScene* scn);

  // graphical user interface elements for agent and task
  void GUI(mjUI& ui);

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

  // return horizon (continuous time)
  double Horizon() const;

  // render plots
  void PlotShow(mjrRect* rect, mjrContext* con);

  // returns all task names, joined with '\n' characters
  std::string GetTaskNames() const { return task_names_; }
  int GetTaskIdByName(std::string_view name) const;
  std::string GetTaskXmlPath(int id) const { return tasks_[id]->XmlPath(); }

  mjpc::Planner& ActivePlanner() const { return *planners_[planner_]; }
  mjpc::State& ActiveState() const { return *states_[state_]; }
  Task* ActiveTask() const { return tasks_[active_task_id_].get(); }
  int GetActionDim() { return model_->nu; }
  mjModel* GetModel() { return model_; }

  void SetTaskList(std::vector<std::shared_ptr<Task>> tasks);
  void SetState(const mjData* data);
  void SetTaskByIndex(int id) { active_task_id_ = id; }

  int max_threads() const { return max_threads_;}

  // status flags, logically should be bool, but mjUI needs int pointers
  int plan_enabled;
  int action_enabled;
  int visualize_enabled;
  int allocate_enabled;
  int plot_enabled;
  int gui_task_id = 0;

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

  std::vector<std::shared_ptr<Task>> tasks_;
  int active_task_id_ = 0;

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
