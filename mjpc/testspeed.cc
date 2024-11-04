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

#include "mjpc/testspeed.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>

#include "mjpc/agent.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"
#include "mjpc/tasks/tasks.h"

namespace mjpc {

namespace {
Task* task;
void residual_callback(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    task->Residual(model, data, data->sensordata);
  }
}
}  // namespace

// Run synchronous planning, print timing info,return 0 if nothing failed.
double SynchronousPlanningCost(std::string task_name, int planner_thread_count,
                               int steps_per_planning_iteration,
                               double total_time) {
  std::cout << "Test MJPC Speed: " << task_name << "\n";
  std::cout << " MuJoCo version " << mj_versionString() << "\n";
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have different versions");
  }
  std::cout << " Hardware threads:  " << NumAvailableHardwareThreads() << "\n";

  Agent agent;
  agent.SetTaskList(GetTasks());
  agent.gui_task_id = agent.GetTaskIdByName(task_name);
  if (agent.gui_task_id == -1) {
    std::cerr << "Invalid --task flag: '" << task_name
              << "'. Valid values:\n";
    std::cerr << agent.GetTaskNames();
    return -1;
  }
  Agent::LoadModelResult load_model = agent.LoadModel();
  mjModel* model = load_model.model.get();
  if (!model) {
    std::cerr << load_model.error << "\n";
    return -1;
  }
  mjData* data = mj_makeData(model);

  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) {
    std::cout << "home_id: " << home_id << "\n";
    mj_resetDataKeyframe(model, data, home_id);
  }
  mj_forward(model, data);

  // the planner and its initial configuration is set in the XML
  agent.estimator_enabled = false;
  agent.Initialize(model);
  agent.Allocate();
  agent.Reset(data->ctrl);
  agent.plan_enabled = true;

  // make task available for global callback:
  task = agent.ActiveTask();
  mjcb_sensor = &residual_callback;

  std::cout << " Planning threads:  " << planner_thread_count << "\n";
  ThreadPool pool(planner_thread_count);

  int total_steps = ceil(total_time / model->opt.timestep);
  int current_time = 0;
  double total_cost = 0;
  auto loop_start = std::chrono::steady_clock::now();
  for (int i = 0; i < total_steps; i++) {
    agent.ActiveTask()->Transition(model, data);
    agent.state.Set(model, data);

    agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, agent.state.state().data(),
        agent.state.time(), /*use_previous=*/false);
    mj_step(model, data);
    double cost = agent.ActiveTask()->CostValue(data->sensordata);
    total_cost += cost;

    if (i % steps_per_planning_iteration == 0) { agent.PlanIteration(&pool); }

    if (floor(data->time) > current_time) {
      current_time++;
      std::cout << "sim time: " << current_time << ", cost: " << cost << "\n";
    }
  }
  auto wall_run_time = std::chrono::duration_cast<std::chrono::microseconds>(
                            std::chrono::steady_clock::now() - loop_start)
                            .count() /
                        1e6;
  std::cout << "Total wall time ("
            << (int)ceil(total_steps / steps_per_planning_iteration)
            << " planning steps): " << wall_run_time << " s ("
            << total_time / wall_run_time << "x realtime)\n";
  std::cout << "Average cost per step (lower is better): "
            << total_cost / total_steps << "\n";

  mj_deleteData(data);
  mjcb_sensor = nullptr;
  return total_cost;
}
}  // namespace mjpc
