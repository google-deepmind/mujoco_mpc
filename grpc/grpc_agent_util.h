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

#ifndef GRPC_GRPC_GRPC_AGENT_UTIL_H_
#define GRPC_GRPC_GRPC_AGENT_UTIL_H_

#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>
#include "grpc/agent.pb.h"
#include "mjpc/agent.h"

namespace grpc_agent_util {
grpc::Status GetState(const mjModel* model, const mjData* data,
                      agent::GetStateResponse* response);
grpc::Status SetState(const agent::SetStateRequest* request, mjpc::Agent* agent,
                      const mjModel* model, mjData* data);
grpc::Status GetAction(const agent::GetActionRequest* request,
                       const mjpc::Agent* agent,
                       agent::GetActionResponse* response);
grpc::Status GetCostValuesAndWeights(
    const agent::GetCostValuesAndWeightsRequest* request,
    const mjpc::Agent* agent, const mjModel* model, mjData* data,
    agent::GetCostValuesAndWeightsResponse* response);
grpc::Status Reset(mjpc::Agent* agent, const mjModel* model, mjData* data);
grpc::Status SetTaskParameters(const agent::SetTaskParametersRequest* request,
                               mjpc::Agent* agent);
grpc::Status SetCostWeights(const agent::SetCostWeightsRequest* request,
                            mjpc::Agent* agent);

// set up the task and model on the agent so that the next call to
// agent.LoadModel returns any custom model, or the relevant task model.
grpc::Status InitAgent(mjpc::Agent* agent, const agent::InitRequest* request);
}  // namespace grpc_agent_util

#endif  // GRPC_GRPC_GRPC_AGENT_UTIL_H_
