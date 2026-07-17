// Copyright 2026 DeepMind Technologies Limited
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

#ifndef MJPC_MJPC_STUDIO_UI_H_
#define MJPC_MJPC_STUDIO_UI_H_

#include <functional>
#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/agent.h"

namespace mjpc::studio {

void DrawMjpcUi(mjpc::Agent* agent, const mjModel* model, mjData* data,
                std::function<void(const std::string&)> load_model_cb,
                bool agent_active);

void DrawMjpcPlots(mjpc::Agent* agent, const mjModel* model, mjData* data);

}  // namespace mjpc::studio

#endif  // MJPC_MJPC_STUDIO_UI_H_
