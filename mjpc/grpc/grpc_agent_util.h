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

#ifndef MJPC_MJPC_GRPC_GRPC_AGENT_UTIL_H_
#define MJPC_MJPC_GRPC_GRPC_AGENT_UTIL_H_

#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "mjpc/grpc/agent.pb.h"
#include "mjpc/agent.h"

namespace grpc_agent_util {
grpc::Status GetState(const mjModel* model, const mjData* data,
                      agent::GetStateResponse* response);
grpc::Status SetState(const agent::SetStateRequest* request, mjpc::Agent* agent,
                      const mjModel* model, mjData* data);
grpc::Status GetAction(const agent::GetActionRequest* request,
                       const mjpc::Agent* agent,
                       const mjModel* model, mjData* rollout_data,
                       mjpc::State* rollout_state,
                       agent::GetActionResponse* response);
grpc::Status GetCostValuesAndWeights(
    const agent::GetCostValuesAndWeightsRequest* request,
    const mjpc::Agent* agent, const mjModel* model, mjData* data,
    agent::GetCostValuesAndWeightsResponse* response);
grpc::Status Reset(mjpc::Agent* agent, const mjModel* model, mjData* data);
grpc::Status SetTaskParameters(const agent::SetTaskParametersRequest* request,
                               mjpc::Agent* agent);
grpc::Status GetTaskParameters(const agent::GetTaskParametersRequest* request,
                               mjpc::Agent* agent,
                              agent::GetTaskParametersResponse* response);
grpc::Status SetCostWeights(const agent::SetCostWeightsRequest* request,
                            mjpc::Agent* agent);
grpc::Status SetMode(const agent::SetModeRequest* request, mjpc::Agent* agent);
grpc::Status GetMode(const agent::GetModeRequest* request, mjpc::Agent* agent,
                     agent::GetModeResponse* response);

mjpc::UniqueMjModel LoadModelFromString(std::string_view xml, char* error,
                             int error_size);
mjpc::UniqueMjModel LoadModelFromBytes(std::string_view mjb);

// set up the task and model on the agent so that the next call to
// agent.LoadModel returns any custom model, or the relevant task model.
grpc::Status InitAgent(mjpc::Agent* agent, const agent::InitRequest* request);
}  // namespace grpc_agent_util

#endif  // MJPC_MJPC_GRPC_GRPC_AGENT_UTIL_H_
