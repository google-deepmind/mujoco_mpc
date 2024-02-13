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

#include "mjpc/grpc/ui_agent_service.h"

#include <memory>
#include <string_view>
#include <vector>

#include <absl/synchronization/notification.h>
#include <absl/time/time.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mjui.h>
#include <mujoco/mujoco.h>

#include "mjpc/grpc/agent.pb.h"
#include "mjpc/grpc/grpc_agent_util.h"
#include "mjpc/agent.h"
#include "mjpc/utilities.h"

namespace mjpc::agent_grpc {

using ::agent::GetActionRequest;
using ::agent::GetActionResponse;
using ::agent::GetAllModesRequest;
using ::agent::GetAllModesResponse;
using ::agent::GetBestTrajectoryRequest;
using ::agent::GetBestTrajectoryResponse;
using ::agent::GetResidualsRequest;
using ::agent::GetResidualsResponse;
using ::agent::GetCostValuesAndWeightsRequest;
using ::agent::GetCostValuesAndWeightsResponse;
using ::agent::GetModeRequest;
using ::agent::GetModeResponse;
using ::agent::GetStateRequest;
using ::agent::GetStateResponse;
using ::agent::GetTaskParametersRequest;
using ::agent::GetTaskParametersResponse;
using ::agent::InitRequest;
using ::agent::InitResponse;
using ::agent::PlannerStepRequest;
using ::agent::PlannerStepResponse;
using ::agent::ResetRequest;
using ::agent::ResetResponse;
using ::agent::SetAnythingRequest;
using ::agent::SetAnythingResponse;
using ::agent::SetCostWeightsRequest;
using ::agent::SetCostWeightsResponse;
using ::agent::SetModeRequest;
using ::agent::SetModeResponse;
using ::agent::SetStateRequest;
using ::agent::SetStateResponse;
using ::agent::SetTaskParametersRequest;
using ::agent::SetTaskParametersResponse;
using ::agent::StepRequest;
using ::agent::StepResponse;
using ::mjpc::UniqueMjModel;

grpc::Status UiAgentService::Init(grpc::ServerContext* context,
                                  const InitRequest* request,
                                  InitResponse* response) {
  grpc::Status status = grpc_agent_util::InitAgent(sim_->agent.get(), request);
  if (!status.ok()) {
    return status;
  }
  // fake a UI event where the task changes
  // TODO(nimrod): get rid of this hack
  mjuiItem it = {0};
  it.itemid = 2;
  sim_->agent->TaskEvent(&it, sim_->d, sim_->uiloadrequest, sim_->run);

  // set real time speed
  float desired_percent = 100;
  if (request->real_time_speed()) {
    desired_percent = 100 * request->real_time_speed();
  }
  auto closest = std::min_element(
      std::begin(sim_->percentRealTime), std::end(sim_->percentRealTime),
      [&](float a, float b) {
        return std::abs(a - desired_percent) < std::abs(b - desired_percent);
      });
  sim_->real_time_index =
      std::distance(std::begin(sim_->percentRealTime), closest);
  // wait until the model changes to update rollout_data_
  return RunBeforeStep(
      context, [&](mjpc::Agent* agent, const mjModel* model, mjData* data) {
        rollout_data_.reset(mj_makeData(model));
        return grpc::Status::OK;
      });
}

grpc::Status UiAgentService::GetState(grpc::ServerContext* context,
                                      const GetStateRequest* request,
                                      GetStateResponse* response) {
  return RunBeforeStep(context, [response](mjpc::Agent* agent,
                                           const mjModel* model, mjData* data) {
    return grpc_agent_util::GetState(model, data, response);
  });
}

grpc::Status UiAgentService::SetState(grpc::ServerContext* context,
                                      const SetStateRequest* request,
                                      SetStateResponse* response) {
  return RunBeforeStep(context, [request](mjpc::Agent* agent,
                                          const mjModel* model, mjData* data) {
    return grpc_agent_util::SetState(request, agent, model, data);
  });
}

grpc::Status UiAgentService::GetAction(grpc::ServerContext* context,
                                       const GetActionRequest* request,
                                       GetActionResponse* response) {
  return RunBeforeStep(context, [&, request, response](mjpc::Agent* agent,
                                                       const mjModel* model,
                                                       mjData* data) {
    return grpc_agent_util::GetAction(
        request, agent, model, rollout_data_.get(), &rollout_state_, response);
  });
}

grpc::Status UiAgentService::GetResiduals(
    grpc::ServerContext* context, const GetResidualsRequest* request,
    GetResidualsResponse* response) {
  return RunBeforeStep(
      context, [request, response](mjpc::Agent* agent, const mjModel* model,
                                   mjData* data) {
        return grpc_agent_util::GetResiduals(request, agent, model,
                                             data, response);
      });
}


grpc::Status UiAgentService::GetCostValuesAndWeights(
    grpc::ServerContext* context, const GetCostValuesAndWeightsRequest* request,
    GetCostValuesAndWeightsResponse* response) {
  return RunBeforeStep(
      context, [request, response](mjpc::Agent* agent, const mjModel* model,
                                   mjData* data) {
        return grpc_agent_util::GetCostValuesAndWeights(request, agent, model,
                                                        data, response);
      });
}

grpc::Status UiAgentService::PlannerStep(grpc::ServerContext* context,
                                         const PlannerStepRequest* request,
                                         PlannerStepResponse* response) {
  // in this setup, the planner is async so this RPC doesn't need to do a
  // planning step. instead, enable the planner if it's not on already.
  sim_->agent->plan_enabled = true;
  return grpc::Status::OK;
}

grpc::Status UiAgentService::Step(grpc::ServerContext* context,
                                  const StepRequest* request,
                                  StepResponse* response) {
  // simulation is assumed to be running - do nothing
  return grpc::Status::OK;
}

grpc::Status UiAgentService::Reset(grpc::ServerContext* context,
                                   const ResetRequest* request,
                                   ResetResponse* response) {
  return RunBeforeStep(
      context, [&](mjpc::Agent* agent, const mjModel* model, mjData* data) {
        grpc::Status status = grpc_agent_util::Reset(agent, model, data);
        rollout_data_.reset(mj_makeData(model));
        return status;
      });
}

grpc::Status UiAgentService::SetTaskParameters(
    grpc::ServerContext* context, const SetTaskParametersRequest* request,
    SetTaskParametersResponse* response) {
  return RunBeforeStep(context, [request](mjpc::Agent* agent,
                                          const mjModel* model, mjData* data) {
    return grpc_agent_util::SetTaskParameters(request, agent);
  });
}

grpc::Status UiAgentService::GetTaskParameters(
    grpc::ServerContext* context, const GetTaskParametersRequest* request,
    GetTaskParametersResponse* response) {
  return RunBeforeStep(context, [request, response](mjpc::Agent* agent,
                                          const mjModel* model, mjData* data) {
    return grpc_agent_util::GetTaskParameters(request, agent, response);
  });
}

grpc::Status UiAgentService::SetCostWeights(
    grpc::ServerContext* context, const SetCostWeightsRequest* request,
    SetCostWeightsResponse* response) {
  return RunBeforeStep(context, [request](mjpc::Agent* agent,
                                          const mjModel* model, mjData* data) {
    return grpc_agent_util::SetCostWeights(request, agent);
  });
}

grpc::Status UiAgentService::SetMode(grpc::ServerContext* context,
                                     const SetModeRequest* request,
                                     SetModeResponse* response) {
  return RunBeforeStep(context, [request](mjpc::Agent* agent,
                                          const mjModel* model, mjData* data) {
    return grpc_agent_util::SetMode(request, agent);
  });
}
grpc::Status UiAgentService::GetMode(grpc::ServerContext* context,
                                     const GetModeRequest* request,
                                     GetModeResponse* response) {
  return RunBeforeStep(
      context, [request, response](mjpc::Agent* agent, const mjModel* model,
                                   mjData* data) {
        return grpc_agent_util::GetMode(request, agent, response);
      });
}

grpc::Status UiAgentService::GetAllModes(grpc::ServerContext* context,
                                         const GetAllModesRequest* request,
                                         GetAllModesResponse* response) {
  return RunBeforeStep(
      context, [request, response](mjpc::Agent* agent, const mjModel* model,
                                   mjData* data) {
        return grpc_agent_util::GetAllModes(request, agent, response);
      });
}

grpc::Status UiAgentService::GetBestTrajectory(
    grpc::ServerContext* context, const GetBestTrajectoryRequest* request,
    GetBestTrajectoryResponse* response) {
  // TODO - Implement.
  return {grpc::StatusCode::UNIMPLEMENTED,
          "GetBestTrajectory is not implemented."};
}

grpc::Status UiAgentService::SetAnything(grpc::ServerContext* context,
                                         const SetAnythingRequest* request,
                                         SetAnythingResponse* response) {
  return RunBeforeStep(context, [request, response](mjpc::Agent* agent,
                                                    const mjModel* model,
                                                    mjData* data) {
    return grpc_agent_util::SetAnything(request, agent, model, data, response);
  });
}

namespace {
bool WaitUntilDeadline(const absl::Notification& notification,
                       const grpc::ServerContext* context) {
  absl::Time deadline = absl::FromChrono(context->deadline());
  return notification.WaitForNotificationWithDeadline(deadline);
}
}  // namespace

grpc::Status UiAgentService::RunBeforeStep(const grpc::ServerContext* context,
                                           StatusStepJob job) {
  grpc::Status status = {grpc::StatusCode::UNKNOWN, ""};
  absl::Notification notification;
  sim_->agent->RunBeforeStep(
      [&job, &status, &notification](mjpc::Agent* agent, const mjModel* model,
                                      mjData* data) {
        status = job(agent, model, data);
        notification.Notify();
      });
  if (!WaitUntilDeadline(notification, context)) {
    return {grpc::StatusCode::DEADLINE_EXCEEDED,
            "Timed out while waiting for physics step."};
  }
  return status;
}
}  // namespace mjpc::agent_grpc
