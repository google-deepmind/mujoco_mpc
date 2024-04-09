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

#include "mjpc/agent.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include <absl/container/flat_hash_map.h>
#include <absl/strings/match.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>
#include <absl/strings/strip.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjui.h>
#include <mujoco/mjvisualize.h>
#include <mujoco/mujoco.h>
#include "mjpc/array_safety.h"
#include "mjpc/estimators/include.h"
#include "mjpc/planners/include.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {
namespace mju = ::mujoco::util_mjpc;

namespace {
// ----- agent constants ----- //
inline constexpr double kMinTimeStep = 1.0e-4;
inline constexpr double kMaxTimeStep = 0.1;
inline constexpr double kMinPlanningHorizon = 1.0e-5;
inline constexpr double kMaxPlanningHorizon = 2.5;

// maximum number of actions to plot
const int kMaxActionPlots = 25;

}  // namespace

Agent::Agent(const mjModel* model, std::shared_ptr<Task> task)
    : Agent::Agent() {
  SetTaskList({std::move(task)});
  Initialize(model);
  Allocate();
  Reset();
  PlotInitialize();
  PlotReset();
}

// initialize data, settings, planners, state
void Agent::Initialize(const mjModel* model) {
  // ----- model ----- //
  mjModel* old_model = model_;
  model_ = mj_copyModel(nullptr, model);  // agent's copy of model

  // check for limits on all actuators
  int num_missing = 0;
  for (int i = 0; i < model_->nu; i++) {
    if (!model_->actuator_ctrllimited[i]) {
      num_missing++;
      printf("%s (actuator %i) missing limits\n",
             model_->names + model_->name_actuatoradr[i], i);
    }
  }
  if (num_missing > 0) {
    mju_error("Ctrl limits required for all actuators.\n");
  }

  // planner
  planner_ = GetNumberOrDefault(0, model, "agent_planner");

  // estimator
  estimator_ =
      estimator_enabled ? GetNumberOrDefault(0, model, "estimator") : 0;

  // integrator
  integrator_ =
      GetNumberOrDefault(model->opt.integrator, model, "agent_integrator");

  // planning horizon
  horizon_ = GetNumberOrDefault(0.5, model, "agent_horizon");

  // time step
  timestep_ = GetNumberOrDefault(1.0e-2, model, "agent_timestep");

  // planning steps
  steps_ = mju_max(mju_min(horizon_ / timestep_ + 1, kMaxTrajectoryHorizon), 1);

  active_task_id_ = gui_task_id;
  ActiveTask()->Reset(model);

  // initialize planner
  for (const auto& planner : planners_) {
    planner->Initialize(model_, *ActiveTask());
  }

  // initialize state
  state.Initialize(model);

  // initialize estimator
  if (reset_estimator && estimator_enabled) {
    for (const auto& estimator : estimators_) {
      estimator->Initialize(model_);
      estimator->Reset();
    }
  }

  // initialize estimator data
  ctrl.resize(model->nu);
  sensor.resize(model->nsensordata);

  // status
  plan_enabled = false;
  action_enabled = true;
  visualize_enabled = false;
  allocate_enabled = true;
  plot_enabled = true;

  // cost
  cost_ = 0.0;

  // counter
  count_ = 0;

  // names
  mju::strcpy_arr(this->planner_names_, kPlannerNames);
  mju::strcpy_arr(this->estimator_names_, kEstimatorNames);

  // estimator threads
  estimator_threads_ = estimator_enabled;

  // planner threads
  planner_threads_ =
      std::max(1, NumAvailableHardwareThreads() - 3 - 2 * estimator_threads_);

  // delete the previous model after all the planners have been updated to use
  // the new one.
  if (old_model) {
    mj_deleteModel(old_model);
  }
}

// allocate memory
void Agent::Allocate() {
  // planner
  for (const auto& planner : planners_) {
    planner->Allocate();
  }

  // state
  state.Allocate(model_);

  // set status
  allocate_enabled = false;

  // cost
  terms_.resize(ActiveTask()->num_term * kMaxTrajectoryHorizon);
}

// reset data, settings, planners, state
void Agent::Reset(const double* initial_repeated_action) {
  // planner
  for (const auto& planner : planners_) {
    planner->Reset(kMaxTrajectoryHorizon, initial_repeated_action);
  }

  // state
  state.Reset();

  // estimator
  if (reset_estimator && estimator_enabled) {
    for (const auto& estimator : estimators_) {
      estimator->Reset();
    }
  }

  // cost
  cost_ = 0.0;

  // count
  count_ = 0;

  // cost
  std::fill(terms_.begin(), terms_.end(), 0.0);
}

void Agent::SetState(const mjData* data) {
  state.Set(model_, data);
}

int Agent::GetTaskIdByName(std::string_view name) const {
  for (int i = 0; i < tasks_.size(); i++) {
    if (absl::EqualsIgnoreCase(name, tasks_[i]->Name())) {
      return i;
    }
  }
  return -1;
}

Agent::LoadModelResult Agent::LoadModel() const {
  // if user specified a custom model, use that.
  mjModel* mnew = nullptr;
  constexpr int kErrorLength = 1024;
  char load_error[kErrorLength] = "";

  if (model_override_) {
    mnew = mj_copyModel(nullptr, model_override_.get());
  } else {
    // otherwise use the task's model
    std::string filename = tasks_[gui_task_id]->XmlPath();
    // make sure filename is not empty
    if (filename.empty()) {
      return {};
    }

    if (absl::StrContains(filename, ".mjb")) {
      mnew = mj_loadModel(filename.c_str(), nullptr);
      if (!mnew) {
        mju::strcpy_arr(load_error, "could not load binary model");
      }
    } else {
      mnew = mj_loadXML(filename.c_str(), nullptr, load_error,
                        kErrorLength);
      // remove trailing newline character from load_error
      if (load_error[0]) {
        int error_length = mju::strlen_arr(load_error);
        if (load_error[error_length - 1] == '\n') {
          load_error[error_length - 1] = '\0';
        }
      }
    }
  }
  return {.model = {mnew, mj_deleteModel},
          .error = load_error};
}

void Agent::OverrideModel(UniqueMjModel model) {
  model_override_ = std::move(model);
}

void Agent::SetTaskList(std::vector<std::shared_ptr<Task>> tasks) {
  tasks_ = std::move(tasks);
  std::ostringstream concatenated_task_names;
  for (const auto& task : tasks_) {
    concatenated_task_names << task->Name() << '\n';
  }
  mju::strcpy_arr(task_names_, concatenated_task_names.str().c_str());
}

void Agent::PlanIteration(ThreadPool* pool) {
  // start agent timer
  auto agent_start = std::chrono::steady_clock::now();

  // set agent time and time step
  model_->opt.timestep = timestep_;
  model_->opt.integrator = integrator_;

  // set planning steps
  steps_ =
      mju_max(mju_min(horizon_ / timestep_ + 1, kMaxTrajectoryHorizon), 1);

  // plan
  if (!allocate_enabled) {
    // set state
    ActivePlanner().SetState(state);

    // copy the task's residual function parameters into a new object, which
    // remains constant during planning and doesn't require locking from the
    // rollout threads
    residual_fn_ = ActiveTask()->Residual();

    if (plan_enabled) {
      // planner policy
      ActivePlanner().OptimizePolicy(steps_, *pool);

      // compute time
      agent_compute_time_ =
          std::chrono::duration_cast<std::chrono::microseconds>(
              std::chrono::steady_clock::now() - agent_start)
              .count();

      // counter
      count_ += 1;
    } else {
      // rollout nominal policy
      ActivePlanner().NominalTrajectory(steps_, *pool);

      // set timers
      agent_compute_time_ = 0.0;
    }

    // release the planning residual function
    residual_fn_.reset();
  }
}

// call planner to update nominal policy
void Agent::Plan(std::atomic<bool>& exitrequest,
                 std::atomic<int>& uiloadrequest) {
  // instantiate thread pool
  ThreadPool pool(planner_threads_);

  // main loop
  while (!exitrequest.load()) {
    if (model_ && uiloadrequest.load() == 0) {
      PlanIteration(&pool);
    }
  }  // exitrequest sent -- stop planning
}

void Agent::RunBeforeStep(StepJob job) {
  std::lock_guard<std::mutex> lock(step_jobs_mutex_);
  step_jobs_.push_back(std::move(job));
}

void Agent::ExecuteAllRunBeforeStepJobs(const mjModel* model, mjData* data) {
  while (true) {
      StepJob step_job;
      {
        // only hold the lock while reading from the queue and not while
        // executing the jobs
        std::lock_guard<std::mutex> lock(step_jobs_mutex_);
        if (step_jobs_.empty()) {
          break;
        }
        step_job = std::move(step_jobs_.front());
        step_jobs_.pop_front();
      }
      step_job(this, model, data);
    }
}

int Agent::SetParamByName(std::string_view name, double value) {
  if (absl::StartsWith(name, "residual_")) {
    name = absl::StripPrefix(name, "residual_");
  }
  if (absl::StartsWith(name, "selection_")) {
    mju_warning(
        "SetParamByName should not be used with selection_ parameters. Use "
        "SetSelectionParamByName.");
    return -1;
  }
  int shift = 0;
  for (int i = 0; i < model_->nnumeric; i++) {
    std::string_view numeric_name(model_->names + model_->name_numericadr[i]);
    if (absl::StartsWith(numeric_name, "residual_")) {
      if (absl::EqualsIgnoreCase(absl::StripPrefix(numeric_name, "residual_"),
                                 name)) {
        ActiveTask()->parameters[shift] = value;
        return i;
      } else {
        shift++;
      }
    }
  }
  return -1;
}

int Agent::SetSelectionParamByName(std::string_view name,
                                   std::string_view value) {
  if (absl::StartsWith(name, "residual_select_")) {
    name = absl::StripPrefix(name, "residual_select_");
  }
  if (absl::StartsWith(name, "selection_")) {
    name = absl::StripPrefix(name, "selection_");
  }
  int shift = 0;
  for (int i = 0; i < model_->nnumeric; i++) {
    std::string_view numeric_name(model_->names + model_->name_numericadr[i]);
    if (absl::StartsWith(numeric_name, "residual_select_")) {
      if (absl::EqualsIgnoreCase(
              absl::StripPrefix(numeric_name, "residual_select_"), name)) {
        ActiveTask()->parameters[shift] =
            ResidualParameterFromSelection(model_, name, value);
        return i;
      } else {
        shift++;
      }
    }
  }
  return -1;
}

int Agent::SetWeightByName(std::string_view name, double value) {
  for (int i = 0; i < model_->nsensor && model_->sensor_type[i] == mjSENS_USER;
       i++) {
    std::string_view sensor_name(model_->names + model_->name_sensoradr[i]);
    if (absl::EqualsIgnoreCase(sensor_name, name)) {
      ActiveTask()->weight[i] = value;
      return i;
    }
  }
  return -1;
}

std::vector<std::string> Agent::GetAllModeNames() const {
  char* transition_str = GetCustomTextData(model_, "task_transition");
  if (transition_str) {
    // split concatenated names
    return absl::StrSplit(transition_str, '|', absl::SkipEmpty());
  }
  return {"default_mode"};
}

std::string Agent::GetModeName() const {
  std::vector<std::string> mode_names = GetAllModeNames();
  return mode_names[ActiveTask()->mode];
}

int Agent::SetModeByName(std::string_view name) {
  char* transition_str = GetCustomTextData(model_, "task_transition");
  if (transition_str) {
    std::vector<std::string> mode_names =
        absl::StrSplit(transition_str, '|', absl::SkipEmpty());
    for (int i = 0; i < mode_names.size(); i++) {
      if (mode_names[i] == name) {
        ActiveTask()->mode = i;
        return i;
      }
    }
    return -1;
  }
  if (name == "default_mode") {
    ActiveTask()->mode = 0;
    return 0;
  }
  return -1;
}

// visualize traces in GUI
void Agent::ModifyScene(mjvScene* scn) {
  // if acting is off make all geom colors grayscale
  if (!action_enabled) {
    int cube = mj_name2id(model_, mjOBJ_TEXTURE, "cube");
    int graycube = mj_name2id(model_, mjOBJ_TEXTURE, "graycube");
    for (int i = 0; i < scn->ngeom; i++) {
      mjvGeom* g = scn->geoms + i;
      // skip static and decor geoms
      if (!(g->category & mjCAT_DYNAMIC)) {
        continue;
      }
      // make grayscale
      double rgb_average = (g->rgba[0] + g->rgba[1] + g->rgba[2]) / 3;
      g->rgba[0] = g->rgba[1] = g->rgba[2] = rgb_average;
      // specifically for the hand task, make grayscale cube.
      if (cube > -1 && graycube > -1 && g->texid == cube) {
        g->texid = graycube;
      }
    }
  }

  if (!visualize_enabled) {
    return;
  }

  // color
  float color[4];
  color[0] = 1.0;
  color[1] = 0.0;
  color[2] = 1.0;
  color[3] = 1.0;

  // width
  double width = GetNumberOrDefault(0.015, model_, "agent_policy_width");

  // scratch
  double zero3[3] = {0};
  double zero9[9] = {0};

  // winner
  const Trajectory* winner = ActivePlanner().BestTrajectory();
  if (!winner) {
    return;
  }

  // policy
  for (int i = 0; i < winner->horizon - 1; i++) {
    int num_trace = ActiveTask()->num_trace;
    for (int j = 0; j < num_trace; j++) {
      // check max geoms
      if (scn->ngeom >= scn->maxgeom) {
        printf("max geom!!!\n");
        continue;
      }

      // initialize geometry
      mjv_initGeom(&scn->geoms[scn->ngeom], mjGEOM_CAPSULE, zero3, zero3, zero9,
                  color);

      // make geometry
      mjv_makeConnector(
          &scn->geoms[scn->ngeom], mjGEOM_CAPSULE, width,
          winner->trace[3 * num_trace * i + 3 * j],
          winner->trace[3 * num_trace * i + 1 + 3 * j],
          winner->trace[3 * num_trace * i + 2 + 3 * j],
          winner->trace[3 * num_trace * (i + 1) + 3 * j],
          winner->trace[3 * num_trace * (i + 1) + 1 + 3 * j],
          winner->trace[3 * num_trace * (i + 1) + 2 + 3 * j]);
      // increment number of geometries
      scn->ngeom += 1;
    }
  }

  // sample traces
  ActivePlanner().Traces(scn);
}

// graphical user interface elements for agent and task
void Agent::GUI(mjUI& ui) {
  // ----- task ------ //
  mjuiDef defTask[] = {
      {mjITEM_SECTION, "Task", 1, nullptr, "AP"},
      {mjITEM_CHECKINT, "Reset", 2, &ActiveTask()->reset, " #459"},
      {mjITEM_CHECKINT, "Visualize", 2, &ActiveTask()->visualize, ""},
      {mjITEM_SELECT, "Model", 1, &gui_task_id, ""},
      {mjITEM_SLIDERNUM, "Risk", 1, &ActiveTask()->risk, "-1 1"},
      {mjITEM_SEPARATOR, "Weights", 1},
      {mjITEM_END}};

  // task names
  mju::strcpy_arr(defTask[3].other, task_names_);
  mjui_add(&ui, defTask);

  // norm weights
  if (ActiveTask()->num_term) {
    mjuiDef defNormWeight[kMaxCostTerms + 1];
    for (int i = 0; i < ActiveTask()->num_term; i++) {
      // element
      defNormWeight[i] = {mjITEM_SLIDERNUM, "weight", 2,
                          DataAt(ActiveTask()->weight, i), "0 1"};

      // name
      mju::strcpy_arr(defNormWeight[i].name,
                      model_->names + model_->name_sensoradr[i]);

      // limits
      double* s = model_->sensor_user + i * model_->nuser_sensor;
      mju::sprintf_arr(defNormWeight[i].other, "%f %f", s[2], s[3]);
    }

    defNormWeight[ActiveTask()->num_term] = {mjITEM_END};
    mjui_add(&ui, defNormWeight);
  }

  // residual parameters
  int parameter_shift = (ActiveTask()->parameters.empty() ? 0 : 1);
  mjuiDef defFeatureParameters[kMaxCostTerms + 2];
  if (parameter_shift > 0) {
    defFeatureParameters[0] = {mjITEM_SEPARATOR, "Parameters", 1};
  }
  for (int i = 0; i < ActiveTask()->parameters.size(); i++) {
    defFeatureParameters[i + parameter_shift] = {
        mjITEM_SLIDERNUM, "residual", 2, DataAt(ActiveTask()->parameters, i),
        "0 1"};
  }

  absl::flat_hash_map<std::string, std::vector<std::string>> selections =
      ResidualSelectionLists(model_);

  int shift = 0;
  for (int i = 0; i < model_->nnumeric; i++) {
    const char* name = model_->names + model_->name_numericadr[i];
    if (absl::StartsWith(name, "residual_select_")) {
      std::string_view list_name = absl::StripPrefix(name, "residual_select_");
      if (auto it = selections.find(list_name); it != selections.end()) {
        // insert a dropdown list
        mjuiDef* uiItem = &defFeatureParameters[shift + parameter_shift];
        uiItem->type = mjITEM_SELECT;
        mju::strcpy_arr(uiItem->name, list_name.data());
        mju::strcpy_arr(uiItem->other, absl::StrJoin(it->second, "\n").c_str());

        // note: uiItem.pdata is pointing at a double in parameters,
        // but mjITEM_SELECT is going to treat is as an int. the
        // ResidualSelection and DefaultResidualSelection functions hide the
        // necessary casting when reading such values.

      } else {
        mju_error_s("Selection list not found for %s", name);
        return;
      }
      shift += 1;
    } else if (absl::StartsWith(name, "residual_")) {
      mjuiDef* uiItem = &defFeatureParameters[shift + parameter_shift];
      // name
      mju::strcpy_arr(uiItem->name, model_->names + model_->name_numericadr[i] +
                                        std::strlen("residual_"));
      // limits
      if (model_->numeric_size[i] == 3) {
        mju::sprintf_arr(uiItem->other, "%f %f",
                         model_->numeric_data[model_->numeric_adr[i] + 1],
                         model_->numeric_data[model_->numeric_adr[i] + 2]);
      }
      shift += 1;
    }
  }
  defFeatureParameters[ActiveTask()->parameters.size() + parameter_shift] = {
      mjITEM_END};
  mjui_add(&ui, defFeatureParameters);

  // transition
  char* names = GetCustomTextData(model_, "task_transition");

  if (names) {
    mjuiDef defTransition[] = {
        {mjITEM_SEPARATOR, "Modes", 1},
        {mjITEM_RADIO, "", 1, &ActiveTask()->mode, ""},
        {mjITEM_END},
    };

    // concatenate names
    int len = strlen(names);
    std::string str;
    for (int i = 0; i < len; i++) {
      if (names[i] == '|') {
        str.push_back('\n');
      } else {
        str.push_back(names[i]);
      }
    }

    // update buttons
    mju::strcpy_arr(defTransition[1].other, str.c_str());

    // set tolerance limits
    mju::sprintf_arr(defTransition[2].other, "%f %f", 0.0, 1.0);

    mjui_add(&ui, defTransition);
  }

  // ----- agent ----- //
  mjuiDef defAgent[] = {{mjITEM_SECTION, "Agent", 1, nullptr, "AP"},
                        {mjITEM_BUTTON, "Reset", 2, nullptr, " #459"},
                        {mjITEM_SELECT, "Planner", 2, &planner_, ""},
                        {mjITEM_SELECT, "Estimator", 2, &estimator_, ""},
                        {mjITEM_CHECKINT, "Plan", 2, &plan_enabled, ""},
                        {mjITEM_CHECKINT, "Action", 2, &action_enabled, ""},
                        {mjITEM_CHECKINT, "Plots", 2, &plot_enabled, ""},
                        {mjITEM_CHECKINT, "Traces", 2, &visualize_enabled, ""},
                        {mjITEM_SEPARATOR, "Agent Settings", 1},
                        {mjITEM_SLIDERNUM, "Horizon", 2, &horizon_, "0 1"},
                        {mjITEM_SLIDERNUM, "Timestep", 2, &timestep_, "0 1"},
                        {mjITEM_SELECT, "Integrator", 2, &integrator_,
                         "Euler\nRK4\nImplicit\nImplicitFast"},
                        {mjITEM_SEPARATOR, "Planner Settings", 1},
                        {mjITEM_END}};

  // planner names
  mju::strcpy_arr(defAgent[2].other, planner_names_);

  // estimator names
  if (!mjpc::GetCustomNumericData(model_, "estimator") || !estimator_enabled) {
    mju::strcpy_arr(defAgent[3].other, "Ground Truth");
  } else {
    mju::strcpy_arr(defAgent[3].other, estimator_names_);
  }

  // set planning horizon slider limits
  mju::sprintf_arr(defAgent[9].other, "%f %f", kMinPlanningHorizon,
                   kMaxPlanningHorizon);

  // set time step limits
  mju::sprintf_arr(defAgent[10].other, "%f %f", kMinTimeStep, kMaxTimeStep);

  // add agent
  mjui_add(&ui, defAgent);

  // planner
  ActivePlanner().GUI(ui);

  // estimator
  if (ActiveEstimatorIndex() > 0) {
    ActiveEstimator().GUI(ui);
  }
}

// task-based GUI event
void Agent::TaskEvent(mjuiItem* it, mjData* data,
                      std::atomic<int>& uiloadrequest, int& run) {
  switch (it->itemid) {
    case 0:  // task reset
      ActiveTask()->Reset(model_);
      ActiveTask()->reset = 0;
      break;
    case 2:  // task switch
      // the GUI changed the value of gui_task_id, but it's unsafe to switch
      // tasks now.
      // turn off agent and traces
      plan_enabled = false;
      action_enabled = false;
      visualize_enabled = false;
      ActiveTask()->visualize = 0;
      ActiveTask()->reset = 0;
      allocate_enabled = true;
      // request model loading
      uiloadrequest.fetch_add(1);
      break;
  }
}

// agent-based GUI event
void Agent::AgentEvent(mjuiItem* it, mjData* data,
                       std::atomic<int>& uiloadrequest, int& run) {
  switch (it->itemid) {
    case 0:  // reset
      if (model_) {
        this->Reset();
        this->PlotInitialize();
        this->PlotReset();
      }
      break;
    case 1:  // planner change
      if (model_) {
        // reset plots
        this->PlotInitialize();
        this->PlotReset();

        // reset agent
        uiloadrequest.fetch_sub(1);
      }
      break;
    case 2:  // estimator change
      // check for estimators
      if (!GetCustomNumericData(model_, "estimator") || !estimator_enabled) {
        estimator_ = 0;
        break;
      }
      // reset
      if (model_) {
        // reset plots
        this->PlotInitialize();
        this->PlotReset();

        // reset estimator
        ActiveEstimator().Reset(data);

        // reset agent
        reset_estimator = false;     // skip estimator reset
        uiloadrequest.fetch_sub(1);  // reset
        reset_estimator = true;      // restore estimator reset
      }
      break;
    case 4:  // controller on/off
      if (model_) {
        mju_zero(data->ctrl, model_->nu);
      }
  }
}

// agent-based GUI event
void Agent::EstimatorEvent(mjuiItem* it, mjData* data,
                           std::atomic<int>& uiloadrequest, int& run) {
  switch (it->itemid) {
    case 0:  // reset estimator
      if (model_) {
        this->ActiveEstimator().Reset(data);
        this->PlotInitialize();
        this->PlotReset();
      }
      break;
  }
}

// initialize plots
void Agent::PlotInitialize() {
  // set figures to default
  mjv_defaultFigure(&plots_.cost);
  mjv_defaultFigure(&plots_.action);
  mjv_defaultFigure(&plots_.planner);
  mjv_defaultFigure(&plots_.timer);

  // don't rescale axes
  plots_.cost.flg_extend = 0;
  plots_.action.flg_extend = 0;
  plots_.planner.flg_extend = 0;
  plots_.timer.flg_extend = 0;

  // title
  mju::strcpy_arr(plots_.cost.title, "Objective");
  mju::strcpy_arr(plots_.action.title, "Actions");
  mju::strcpy_arr(plots_.planner.title, "Agent (log10)");
  mju::strcpy_arr(plots_.timer.title, "CPU time (msec)");

  // x-labels
  mju::strcpy_arr(plots_.action.xlabel, "Time");
  mju::strcpy_arr(plots_.timer.xlabel, "Iteration");

  // y-tick number formats
  mju::strcpy_arr(plots_.cost.yformat, "%.2f");
  mju::strcpy_arr(plots_.action.yformat, "%.2f");
  mju::strcpy_arr(plots_.planner.yformat, "%.2f");
  mju::strcpy_arr(plots_.timer.yformat, "%.2f");

  // ----- colors ----- //

  // history costs
  plots_.cost.linergb[0][0] = 1.0f;
  plots_.cost.linergb[0][1] = 1.0f;
  plots_.cost.linergb[0][2] = 1.0f;

  // current line
  plots_.cost.linergb[1][0] = 1.0f;
  plots_.cost.linergb[1][1] = 0.647f;
  plots_.cost.linergb[1][2] = 0.0f;

  // policy line
  plots_.cost.linergb[2][0] = 1.0f;
  plots_.cost.linergb[2][1] = 0.647f;
  plots_.cost.linergb[2][2] = 0.0f;

  // best cost
  plots_.cost.linergb[3][0] = 1.0f;
  plots_.cost.linergb[3][1] = 1.0f;
  plots_.cost.linergb[3][2] = 1.0f;
  int num_term = ActiveTask()->num_term;
  for (int i = 0; i < num_term; i++) {
    int nclr = kNCostColors;
    // history
    plots_.cost.linergb[4 + i][0] = CostColors[i % nclr][0];
    plots_.cost.linergb[4 + i][1] = CostColors[i % nclr][1];
    plots_.cost.linergb[4 + i][2] = CostColors[i % nclr][2];

    // prediction
    plots_.cost.linergb[4 + num_term + i][0] = 0.9 * CostColors[i % nclr][0];
    plots_.cost.linergb[4 + num_term + i][1] = 0.9 * CostColors[i % nclr][1];
    plots_.cost.linergb[4 + num_term + i][2] = 0.9 * CostColors[i % nclr][2];
  }

  // history of control
  int dim_action = mju_min(model_->nu, kMaxActionPlots);

  for (int i = 0; i < dim_action; i++) {
    plots_.action.linergb[i][0] = 0.0f;
    plots_.action.linergb[i][1] = 1.0f;
    plots_.action.linergb[i][2] = 1.0f;
  }

  // best control
  for (int i = 0; i < dim_action; i++) {
    plots_.action.linergb[dim_action + i][0] = 1.0f;
    plots_.action.linergb[dim_action + i][1] = 0.0f;
    plots_.action.linergb[dim_action + i][2] = 1.0f;
  }

  // current line
  plots_.action.linergb[2 * dim_action][0] = 1.0f;
  plots_.action.linergb[2 * dim_action][1] = 0.647f;
  plots_.action.linergb[2 * dim_action][2] = 0.0f;

  // policy line
  plots_.action.linergb[2 * dim_action + 1][0] = 1.0f;
  plots_.action.linergb[2 * dim_action + 1][1] = 0.647f;
  plots_.action.linergb[2 * dim_action + 1][2] = 0.0f;

  // history of agent compute time
  plots_.timer.linergb[0][0] = 1.0f;
  plots_.timer.linergb[0][1] = 1.0f;
  plots_.timer.linergb[0][2] = 1.0f;

  // x-tick labels
  plots_.cost.flg_ticklabel[0] = 0;
  plots_.action.flg_ticklabel[0] = 0;
  plots_.planner.flg_ticklabel[0] = 0;
  plots_.timer.flg_ticklabel[0] = 0;

  // legends

  // grid sizes
  plots_.cost.gridsize[0] = 3;
  plots_.cost.gridsize[1] = 3;
  plots_.action.gridsize[0] = 3;
  plots_.action.gridsize[1] = 3;
  plots_.planner.gridsize[0] = 3;
  plots_.planner.gridsize[1] = 3;
  plots_.timer.gridsize[0] = 3;
  plots_.timer.gridsize[1] = 3;

  // initialize
  for (int j = 0; j < 20; j++) {
    for (int i = 0; i < mjMAXLINEPNT; i++) {
      plots_.planner.linedata[j][2 * i] = static_cast<float>(-i);
      plots_.timer.linedata[j][2 * i] = static_cast<float>(-i);

      // colors
      if (j == 0) continue;
      plots_.planner.linergb[j][0] = CostColors[j][0];
      plots_.planner.linergb[j][1] = CostColors[j][1];
      plots_.planner.linergb[j][2] = CostColors[j][2];

      plots_.timer.linergb[j][0] = CostColors[j][0];
      plots_.timer.linergb[j][1] = CostColors[j][1];
      plots_.timer.linergb[j][2] = CostColors[j][2];
    }
  }
}

// reset plot data to zeros
void Agent::PlotReset() {
  // cost reset
  for (int k = 0; k < 4 + 2 * ActiveTask()->num_term; k++) {
    PlotResetData(&plots_.cost, 1000, k);
  }

  // action reset
  for (int j = 0; j < 2 * mju_min(model_->nu, kMaxActionPlots) + 2; j++) {
    PlotResetData(&plots_.action, 1000, j);
  }

  // compute time reset
  for (int k = 0; k < 20; k++) {
    PlotResetData(&plots_.planner, 100, k);
    PlotResetData(&plots_.timer, 100, k);

    // reset x tick marks
    for (int i = 0; i < mjMAXLINEPNT; i++) {
      plots_.planner.linedata[k][2 * i] = static_cast<float>(-i);
      plots_.timer.linedata[k][2 * i] = static_cast<float>(-i);
    }
  }
}

// return horizon (continuous time)
double Agent::Horizon() const {
  return horizon_;
}

// plot current information
void Agent::Plots(const mjData* data, int shift) {
  if (allocate_enabled) {
    return;
  }

  // time lower bound
  double time_lower_bound = data->time - horizon_ + model_->opt.timestep;

  // winning trajectory
  const Trajectory* winner = ActivePlanner().BestTrajectory();
  if (!winner) {
    return;
  }

  // ----- cost ----- //
  double cost_bounds[2] = {0.0, 1.0};

  // compute current cost
  // residual values are the first entries in sensordata
  const double* residual = data->sensordata;
  cost_ = ActiveTask()->CostValue(residual);

  // compute individual costs
  for (int t = 0; t < winner->horizon; t++) {
    ActiveTask()->CostTerms(
        DataAt(terms_, t * ActiveTask()->num_term),
        DataAt(winner->residual, t * ActiveTask()->num_residual));
  }

  // shift data
  if (shift) {
    // return
    PlotUpdateData(&plots_.cost, cost_bounds, data->time, cost_, 1000, 0, 1, 1,
                   time_lower_bound);
  }

  // predicted costs
  PlotData(&plots_.cost, cost_bounds, winner->times.data(),
           winner->costs.data(), 1, 1, winner->horizon, 3, time_lower_bound);

  // ranges
  plots_.cost.range[0][0] = data->time - horizon_ + model_->opt.timestep;
  plots_.cost.range[0][1] = data->time + horizon_ - model_->opt.timestep;
  plots_.cost.range[1][0] = cost_bounds[0];
  plots_.cost.range[1][1] = cost_bounds[1];

  // legend
  mju::strcpy_arr(plots_.cost.linename[0], "Total Cost");

  // plot costs
  for (int k = 0; k < ActiveTask()->num_term; k++) {
    // current residual
    if (shift) {
      PlotUpdateData(&plots_.cost, cost_bounds, data->time, terms_[k], 1000,
                     4 + k, 1, 1, time_lower_bound);
    }
    // legend
    mju::strcpy_arr(plots_.cost.linename[4 + ActiveTask()->num_term + k],
                    model_->names + model_->name_sensoradr[k]);
  }

  // predicted residual
  PlotData(&plots_.cost, cost_bounds, winner->times.data(), terms_.data(),
           ActiveTask()->num_term, ActiveTask()->num_term, winner->horizon,
           4 + ActiveTask()->num_term, time_lower_bound);

  // vertical lines at current time and agent time
  PlotVertical(&plots_.cost, data->time, cost_bounds[0], cost_bounds[1], 10, 1);
  PlotVertical(&plots_.cost,
               (winner->times[0] > 0.0 ? winner->times[0] : data->time),
               cost_bounds[0], cost_bounds[1], 10, 2);

  // ----- action ----- //
  double action_bounds[2] = {-1.0, 1.0};

  int dim_action = mju_min(model_->nu, kMaxActionPlots);

  // shift data
  if (shift) {
    // agent history
    for (int j = 0; j < dim_action; j++) {
      PlotUpdateData(&plots_.action, action_bounds, data->time, data->ctrl[j],
                     1000, j, 1, 1, time_lower_bound);
    }
  }

  // agent actions
  PlotData(&plots_.action, action_bounds, winner->times.data(),
           winner->actions.data(), model_->nu, dim_action, winner->horizon,
           dim_action, time_lower_bound);

  // set final action for visualization
  for (int j = 0; j < dim_action; j++) {
    // set data
    if (winner->horizon > 1) {
      plots_.action.linedata[dim_action + j][2 * (winner->horizon - 1) + 1] =
          winner->actions[(winner->horizon - 2) * model_->nu + j];
    } else {
      plots_.action.linedata[dim_action + j][2 * (winner->horizon - 1) + 1] = 0;
    }
  }

  // vertical lines at current time and agent time
  PlotVertical(&plots_.action, data->time, action_bounds[0], action_bounds[1],
               10, 2 * dim_action);
  PlotVertical(&plots_.action,
               (winner->times[0] > 0.0 ? winner->times[0] : data->time),
               action_bounds[0], action_bounds[1], 10, 2 * dim_action + 1);

  // ranges
  plots_.action.range[0][0] = data->time - horizon_ + model_->opt.timestep;
  plots_.action.range[0][1] = data->time + horizon_ - model_->opt.timestep;
  plots_.action.range[1][0] = action_bounds[0];
  plots_.action.range[1][1] = action_bounds[1];

  // legend
  mju::strcpy_arr(plots_.action.linename[0], "History");
  mju::strcpy_arr(plots_.action.linename[dim_action], "Prediction");

  // ----- planner ----- //

  // ranges
  plots_.planner.range[0][0] = -100;
  plots_.planner.range[0][1] = 0;
  plots_.planner.range[1][0] = -6.0;
  plots_.planner.range[1][1] = 6.0;
  plots_.timer.range[0][0] = -100;
  plots_.timer.range[0][1] = 0;
  plots_.timer.range[1][0] = 0.0;

  // skip if planning off
  if (!plan_enabled) return;

  // planner-specific plotting
  int planner_shift[2] {0, 0};
  ActivePlanner().Plots(&plots_.planner, &plots_.timer, 0, 1, plan_enabled,
                        planner_shift);

  // estimator-specific plotting
  if (ActiveEstimatorIndex() > 0) {
    ActiveEstimator().Plots(&plots_.planner, &plots_.timer, planner_shift[0],
                            planner_shift[1] + 1, plan_enabled, NULL);
  }

  // total (agent) compute time
  double timer_bounds[2] = {0.0, 1.0};
  PlotUpdateData(&plots_.timer, timer_bounds, plots_.timer.linedata[0][0] + 1,
                 1.0e-3 * agent_compute_time_, 100, 0, 0, 1, -100);

  // legend
  mju::strcpy_arr(plots_.timer.linename[0], "Total");

  // update timer range
  plots_.timer.range[1][1] = timer_bounds[1];
}

// render plots
void Agent::PlotShow(mjrRect* rect, mjrContext* con) {
  int num_sections = 4;
  mjrRect viewport = {rect->left + rect->width - rect->width / num_sections,
                      rect->bottom, rect->width / num_sections,
                      rect->height / num_sections};
  mjr_figure(viewport, &plots_.timer, con);
  viewport.bottom += rect->height / num_sections;
  mjr_figure(viewport, &plots_.planner, con);
  viewport.bottom += rect->height / num_sections;
  mjr_figure(viewport, &plots_.action, con);
  viewport.bottom += rect->height / num_sections;
  mjr_figure(viewport, &plots_.cost, con);
}

}  // namespace mjpc
