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

// An implementation of the `Agent` gRPC service.

#ifndef MJPC_MJPC_GRPC_AGENT_SERVICE_H_
#define MJPC_MJPC_GRPC_AGENT_SERVICE_H_

#include <memory>
#include <vector>

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include <mjpc/grpc/agent.grpc.pb.h>
#include <mjpc/grpc/agent.pb.h>
#include <mjpc/agent.h>
#include <mjpc/task.h>
#include <mjpc/threadpool.h>
#include <mjpc/utilities.h>

namespace mjpc::agent_grpc {

class AgentService final : public agent::Agent::Service {
 public:
  explicit AgentService(std::vector<std::shared_ptr<mjpc::Task>> tasks,
                        int num_workers = -1)
      : thread_pool_(num_workers == -1 ? mjpc::NumAvailableHardwareThreads()
                                       : num_workers),
        tasks_(std::move(tasks)),
        rollout_data_(nullptr, mj_deleteData) {}
  ~AgentService();
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

  grpc::Status GetResiduals(
      grpc::ServerContext* context,
      const agent::GetResidualsRequest* request,
      agent::GetResidualsResponse* response) override;

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

  grpc::Status SetCostWeights(
      grpc::ServerContext* context,
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

  grpc::Status GetAllModes(grpc::ServerContext* context,
                           const agent::GetAllModesRequest* request,
                           agent::GetAllModesResponse* response) override;

  grpc::Status GetBestTrajectory(
      grpc::ServerContext* context,
      const agent::GetBestTrajectoryRequest* request,
      agent::GetBestTrajectoryResponse* response) override;

  grpc::Status SetAnything(grpc::ServerContext* context,
                           const agent::SetAnythingRequest* request,
                           agent::SetAnythingResponse* response) override;

 private:
  bool Initialized() const { return data_ != nullptr; }

  mjpc::ThreadPool thread_pool_;
  mjpc::Agent agent_;
  std::vector<std::shared_ptr<mjpc::Task>> tasks_;
  mjData* data_ = nullptr;

  // an mjData instance used for rollouts for action averaging
  mjpc::UniqueMjData rollout_data_;
  mjpc::State rollout_state_;
};

}  // namespace mjpc::agent_grpc

#endif  // MJPC_MJPC_GRPC_AGENT_SERVICE_H_
