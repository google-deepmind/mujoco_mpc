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
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include <absl/functional/any_invocable.h>
#include <mujoco/mujoco.h>
#include "mjpc/estimators/include.h"
#include "mjpc/planners/include.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

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
  Agent()
      : planners_(mjpc::LoadPlanners()), estimators_(mjpc::LoadEstimators()) {}
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
  void Reset(const double* initial_repeated_action = nullptr);

  // single planner iteration
  void PlanIteration(ThreadPool* pool);

  // call planner to update nominal policy
  void Plan(std::atomic<bool>& exitrequest, std::atomic<int>& uiloadrequest);

  using StepJob =
      absl::AnyInvocable<void(Agent*, const mjModel*, mjData*)>;

  // runs a callback before the next physics step, on the physics thread
  void RunBeforeStep(StepJob job);

  // executes all the callbacks added by RunBeforeStep. should be called on the
  // physics thread
  void ExecuteAllRunBeforeStepJobs(const mjModel* model, mjData* data);

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

  // estimator-based GUI event
  void EstimatorEvent(mjuiItem* it, mjData* data,
                      std::atomic<int>& uiloadrequest, int& run);

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

  // load the latest task model, based on GUI settings
  struct LoadModelResult {
    UniqueMjModel model{nullptr, mj_deleteModel};
    std::string error;
  };
  LoadModelResult LoadModel() const;

  // Sets a custom model (not from the task), to be returned by the next
  // call to LoadModel. Passing nullptr model clears the override and will
  // return the normal task's model.
  void OverrideModel(UniqueMjModel model = {nullptr, mj_deleteModel});

  mjpc::Planner& ActivePlanner() const { return *planners_[planner_]; }
  mjpc::Estimator& ActiveEstimator() const { return *estimators_[estimator_]; }
  int ActiveEstimatorIndex() const { return estimator_; }
  double ComputeTime() const { return agent_compute_time_; }
  Task* ActiveTask() const { return tasks_[active_task_id_].get(); }
  // a residual function that can be used from trajectory rollouts. must only
  // be used from trajectory rollout threads (no locking).
  const ResidualFn* PlanningResidual() const {
    return residual_fn_.get();
  }
  bool IsPlanningModel(const mjModel* model) const {
    return model == model_;
  }
  int PlanSteps() const { return steps_; }
  int GetActionDim() const { return model_->nu; }
  mjModel* GetModel() { return model_; }
  const mjModel* GetModel() const { return model_; }

  void SetTaskList(std::vector<std::shared_ptr<Task>> tasks);
  void SetState(const mjData* data);
  void SetTaskByIndex(int id) { active_task_id_ = id; }
  // returns param index, or -1 if not found.
  int SetParamByName(std::string_view name, double value);
  // returns param index, or -1 if not found.
  int SetSelectionParamByName(std::string_view name, std::string_view value);
  // returns weight index, or -1 if not found.
  int SetWeightByName(std::string_view name, double value);
  // returns mode index, or -1 if not found.
  int SetModeByName(std::string_view name);

  std::vector<std::string> GetAllModeNames() const;
  std::string GetModeName() const;

  // threads
  int planner_threads() const { return planner_threads_;}
  int estimator_threads() const { return estimator_threads_;}

  // status flags, logically should be bool, but mjUI needs int pointers
  int plan_enabled;
  int action_enabled;
  int visualize_enabled;
  int allocate_enabled;
  int plot_enabled;
  int gui_task_id = 0;

  // state
  mjpc::State state;

  // estimator
  std::vector<double> sensor;
  std::vector<double> ctrl;
  bool reset_estimator = true;
  bool estimator_enabled = false;

 private:
  // model
  mjModel* model_ = nullptr;

  UniqueMjModel model_override_ = {nullptr, mj_deleteModel};

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

  // residual function for the active task, updated once per planning iteration
  std::unique_ptr<ResidualFn> residual_fn_;

  // planners
  std::vector<std::unique_ptr<mjpc::Planner>> planners_;
  int planner_;

  // estimators
  std::vector<std::unique_ptr<mjpc::Estimator>> estimators_;
  int estimator_;

  // task queue for RunBeforeStep
  std::mutex step_jobs_mutex_;
  std::deque<StepJob> step_jobs_;

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
  char estimator_names_[1024];

  // plots
  AgentPlots plots_;

  // max threads for planning
  int planner_threads_;

  // max threads for estimation
  int estimator_threads_;

  // differentiable planning model
  bool differentiable_;
  std::vector<double> jnt_solimp_;
  std::vector<double> geom_solimp_;
  std::vector<double> pair_solimp_;
};

}  // namespace mjpc

#endif  // MJPC_AGENT_H_
