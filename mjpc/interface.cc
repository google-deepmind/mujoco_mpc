// Copyright 2021 DeepMind Technologies Limited
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

#include <memory>
#include <string>
#include <mujoco/mujoco.h>
#include "agent.h"
#include "task.h"
#include "tasks/tasks.h"
#include "threadpool.h"
#include "utilities.h"


namespace mjpc {
class AgentRunner{
 public:
  ~AgentRunner();
  explicit AgentRunner(mjModel* model);
  void Step(mjData* data);
  void Residuals(const mjModel* model, mjData* data);
 private:
  mjpc::Agent agent_;
  mjpc::ThreadPool agent_plan_pool_;
  std::atomic_bool exit_request_ = false;
  std::atomic_int ui_load_request_ = 0;
};

AgentRunner::AgentRunner(mjModel* model) : agent_plan_pool_(1) {
  int task_index = GetNumberOrDefault(0, model, "interface_task_index");
  auto task = mjpc::kTasks[task_index];
  agent_.Initialize(model, "", "", task.residual, task.transition);
  agent_.Allocate();
  agent_.Reset();
  agent_.plan_enabled = true;
  agent_.action_enabled = true;
  agent_.visualize_enabled = false;
  agent_.plot_enabled = false;
  exit_request_.store(false);
  agent_plan_pool_.Schedule(
      [this]() { agent_.Plan(exit_request_, ui_load_request_); });
}

AgentRunner::~AgentRunner() {
  exit_request_.store(true);  // ask the planner threadpool to stop
  agent_plan_pool_.WaitCount(1);  // make sure it's stopped
}

void AgentRunner::Step(mjData* data) {
  agent_.SetState(data);
  agent_.ActivePlanner().ActionFromPolicy(
    data->ctrl, &agent_.ActiveState().state()[0], agent_.ActiveState().time());
}

void AgentRunner::Residuals(const mjModel* model, mjData* data) {
  agent_.task().Residuals(model, data, data->sensordata);
}
}  // namespace mjpc

namespace {
mjpc::AgentRunner* runner = nullptr;

// not exposed to Unity, "extern C" is for MuJoco's callback assignment:
extern "C" void residuals_sensor_callback(const mjModel* model, mjData* data,
                                          int stage) {
  if (stage == mjSTAGE_ACC) {
    runner->Residuals(model, data);
  }
}
}  // namespace


extern "C" void destroy_policy() {
  if (runner != nullptr) {
    delete runner;
    runner = nullptr;
  }
}

extern "C" void create_policy(mjModel* model) {
  if (!mjcb_sensor) {
    mjcb_sensor = residuals_sensor_callback;
  }
  destroy_policy();
  runner = new mjpc::AgentRunner(model);
}

extern "C" void step_policy(mjData* data) {
  if (runner != nullptr) {
    runner->Step(data);
  }
}
