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

#ifndef MJPC_MJPC_INTERFACE_H_
#define MJPC_MJPC_INTERFACE_H_

#include <memory>
#include <string>
#include <vector>
#include <mujoco/mujoco.h>
#include "mjpc/agent.h"
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"


namespace mjpc {
class AgentRunner{
 public:
  ~AgentRunner();
  explicit AgentRunner(const mjModel* model, std::shared_ptr<Task> task);
  void Step(mjData* data);
  void Residual(const mjModel* model, mjData* data);
 private:
  Agent agent_;
  ThreadPool agent_plan_pool_;
  std::atomic_bool exit_request_ = false;
  std::atomic_int ui_load_request_ = 0;
};

}  // namespace mjpc

extern "C" void destroy_policy();
extern "C" void create_policy_from_task_id(const mjModel* model, int task_id);
extern "C" void create_policy(const mjModel* model,
                              std::shared_ptr<mjpc::Task> task);
extern "C" void step_policy(mjData* data);

#endif  // MJPC_MJPC_INTERFACE_H_
