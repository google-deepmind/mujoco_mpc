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

#ifndef MJPC_MJPC_GRPC_UI_AGENT_SERVICE_H_
#define MJPC_MJPC_GRPC_UI_AGENT_SERVICE_H_

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include <mjpc/grpc/agent.grpc.pb.h>
#include <mjpc/grpc/agent.pb.h>
#include <mjpc/simulate.h>  // mjpc fork
#include <mjpc/utilities.h>

namespace mjpc::agent_grpc {

// An AgentService implementation that connects to a running instance of the
// MJPC UI.
class UiAgentService final : public agent::Agent::Service {
 public:
  explicit UiAgentService(mujoco::Simulate* sim)
      : sim_(sim), rollout_data_(nullptr, mj_deleteData) {}

  grpc::Status Init(grpc::ServerContext* context,
                    const agent::InitRequest* request,
                    agent::InitResponse* response) override;

  grpc::Status GetState(grpc::ServerContext* context,
                        const agent::GetStateRequest* request,
                        agent::GetStateResponse* response) override;

  grpc::Status SetState(grpc::ServerContext* context,
                        const agent::SetStateRequest* request,
                        agent::SetStateResponse* response) override;

  grpc::Status GetAction(grpc::ServerContext* context,
                         const agent::GetActionRequest* request,
                         agent::GetActionResponse* response) override;

  grpc::Status GetCostValuesAndWeights(
      grpc::ServerContext* context,
      const agent::GetCostValuesAndWeightsRequest* request,
      agent::GetCostValuesAndWeightsResponse* response) override;

  grpc::Status PlannerStep(grpc::ServerContext* context,
                           const agent::PlannerStepRequest* request,
                           agent::PlannerStepResponse* response) override;

  grpc::Status Step(grpc::ServerContext* context,
                    const agent::StepRequest* request,
                    agent::StepResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const agent::ResetRequest* request,
                     agent::ResetResponse* response) override;

  grpc::Status SetTaskParameters(
      grpc::ServerContext* context,
      const agent::SetTaskParametersRequest* request,
      agent::SetTaskParametersResponse* response) override;

  grpc::Status GetTaskParameters(
      grpc::ServerContext* context,
      const agent::GetTaskParametersRequest* request,
      agent::GetTaskParametersResponse* response) override;

  grpc::Status SetCostWeights(grpc::ServerContext* context,
                              const agent::SetCostWeightsRequest* request,
                              agent::SetCostWeightsResponse* response) override;

  grpc::Status SetMode(
      grpc::ServerContext* context,
      const agent::SetModeRequest* request,
      agent::SetModeResponse* response) override;

  grpc::Status GetMode(
      grpc::ServerContext* context,
      const agent::GetModeRequest* request,
      agent::GetModeResponse* response) override;

 private:
  using StatusStepJob =
      absl::AnyInvocable<grpc::Status(mjpc::Agent*, const mjModel*, mjData*)>;
  // runs a task before the next physics step, on the physics thread, and waits
  // for it to run, up to the deadline of the incoming RPC.
  grpc::Status RunBeforeStep(const grpc::ServerContext* context,
                             StatusStepJob job);

  // Simulate instance owned by the containing binary
  mujoco::Simulate* sim_;

  // an mjData instance used for rollouts for action averaging
  mjpc::UniqueMjData rollout_data_;
  mjpc::State rollout_state_;
};

}  // namespace mjpc::agent_grpc

#endif  // MJPC_MJPC_GRPC_UI_AGENT_SERVICE_H_
