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

#include "grpc/grpc_agent_util.h"

#include <sstream>
#include <string_view>
#include <vector>

#include <absl/log/check.h>
#include <absl/status/status.h>
#include <absl/strings/match.h>
#include <absl/strings/strip.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>
#include "grpc/agent.pb.h"
#include "mjpc/agent.h"
#include "mjpc/task.h"

namespace grpc_agent_util {

using ::agent::GetActionRequest;
using ::agent::GetActionResponse;
using ::agent::GetStateResponse;
using ::agent::SetCostWeightsRequest;
using ::agent::SetStateRequest;
using ::agent::SetTaskParametersRequest;

grpc::Status GetState(const mjModel* model, const mjData* data,
                      GetStateResponse* response) {
  agent::State* output_state = response->mutable_state();

  output_state->set_time(data->time);
  for (int i = 0; i < model->nq; i++) {
    output_state->add_qpos(data->qpos[i]);
  }
  for (int i = 0; i < model->nv; i++) {
    output_state->add_qvel(data->qvel[i]);
  }
  for (int i = 0; i < model->na; i++) {
    output_state->add_act(data->act[i]);
  }
  for (int i = 0; i < model->nmocap * 3; i++) {
    output_state->add_mocap_pos(data->mocap_pos[i]);
  }
  for (int i = 0; i < model->nmocap * 4; i++) {
    output_state->add_mocap_quat(data->mocap_quat[i]);
  }
  for (int i = 0; i < model->nuserdata; i++) {
    output_state->add_userdata(data->userdata[i]);
  }

  return grpc::Status::OK;
}

namespace {
absl::Status CheckSize(std::string_view name, int model_size, int vector_size) {
  std::ostringstream error_string;
  if (model_size != vector_size) {
    error_string << "expected " << name << " size " << model_size << ", got "
                 << vector_size;
    return absl::InvalidArgumentError(error_string.str());
  }
  return absl::OkStatus();
}
}  // namespace

#define CHECK_SIZE(name, n1, n2) \
{ \
  auto expr = (CheckSize(name, n1, n2)); \
if (!(expr).ok()) { \
  return grpc::Status( \
    grpc::StatusCode::INVALID_ARGUMENT, \
    (expr).ToString()); \
  } \
}

grpc::Status SetState(const SetStateRequest* request, mjpc::Agent* agent,
                      const mjModel* model, mjData* data) {
  agent::State state = request->state();

  if (state.has_time()) data->time = state.time();

  if (state.qpos_size() > 0) {
    CHECK_SIZE("qpos", model->nq, state.qpos_size());
    mju_copy(data->qpos, state.qpos().data(), model->nq);
  }

  if (state.qvel_size() > 0) {
    CHECK_SIZE("qvel", model->nv, state.qvel_size());
    mju_copy(data->qvel, state.qvel().data(), model->nv);
  }

  if (state.act_size() > 0) {
    CHECK_SIZE("act", model->na, state.act_size());
    mju_copy(data->act, state.act().data(), model->na);
  }

  if (state.mocap_pos_size() > 0) {
    CHECK_SIZE("mocap_pos", model->nmocap * 3, state.mocap_pos_size());
    mju_copy(data->mocap_pos, state.mocap_pos().data(), model->nmocap * 3);
  }

  if (state.mocap_quat_size() > 0) {
    CHECK_SIZE("mocap_quat", model->nmocap * 4, state.mocap_quat_size());
    mju_copy(data->mocap_quat, state.mocap_quat().data(), model->nmocap * 4);
  }

  if (state.userdata_size() > 0) {
    CHECK_SIZE("userdata", model->nuserdata, state.userdata_size());
    mju_copy(data->userdata, state.userdata().data(), model->nuserdata);
  }

  agent->SetState(data);

  return grpc::Status::OK;
}

#undef CHECK_SIZE

grpc::Status GetAction(const GetActionRequest* request,
                       const mjpc::Agent* agent, GetActionResponse* response) {
  int nu = agent->GetActionDim();
  std::vector<double> ret = std::vector<double>(nu);

  double time =
      request->has_time() ? request->time() : agent->ActiveState().time();

  agent->ActivePlanner().ActionFromPolicy(
      ret.data(), &agent->ActiveState().state()[0], time);

  response->mutable_action()->Assign(ret.begin(), ret.end());

  return grpc::Status::OK;
}

grpc::Status Reset(mjpc::Agent* agent, const mjModel* model, mjData* data) {
  agent->Reset();
  // TODO(nimrod): This should be in the agent, no?
  int home_id = mj_name2id(model, mjOBJ_KEY, "home");
  if (home_id >= 0) {
    mj_resetDataKeyframe(model, data, home_id);
  } else {
    mj_resetData(model, data);
  }
  agent->SetState(data);
  return grpc::Status::OK;
}

grpc::Status SetTaskParameters(const SetTaskParametersRequest* request,
                               mjpc::Agent* agent) {
  for (const auto& [name, value] : request->parameters()) {
    switch (value.value_case()) {
      case agent::TaskParameterValue::kNumeric:
        if (agent->SetParamByName(name, value.numeric()) == -1) {
          std::ostringstream error_string;
          error_string << "Parameter " << name
                       << " not found in task.  Available names are:\n";
          auto* agent_model = agent->GetModel();
          for (int i = 0; i < agent_model->nnumeric; i++) {
            std::string_view numeric_name(agent_model->names +
                                          agent_model->name_numericadr[i]);
            if (absl::StartsWith(numeric_name, "residual_") &&
                !absl::StartsWith(numeric_name, "residual_select_")) {
              error_string << absl::StripPrefix(numeric_name, "residual_")
                           << "\n";
            }
          }
          return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                              error_string.str());
        }
        break;
      case agent::TaskParameterValue::kSelection:
        if (agent->SetSelectionParamByName(name, value.selection()) == -1) {
          std::ostringstream error_string;
          error_string << "Parameter " << name
                       << " not found in task.  Available names are:\n";
          auto agent_model = agent->GetModel();
          for (int i = 0; i < agent_model->nnumeric; i++) {
            std::string_view numeric_name(agent_model->names +
                                          agent_model->name_numericadr[i]);
            if (absl::StartsWith(numeric_name, "residual_select_")) {
              error_string << absl::StripPrefix(numeric_name,
                                                "residual_select_")
                           << "\n";
            }
          }
          return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                              error_string.str());
        }
        break;
      default:
        return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                            absl::StrCat("Missing value for parameter ", name));
    }
  }

  return grpc::Status::OK;
}

grpc::Status SetCostWeights(const SetCostWeightsRequest* request,
                            mjpc::Agent* agent) {
  if (request->reset_to_defaults()) {
    agent->ActiveTask()->Reset(agent->GetModel());
  }
  for (const auto& [name, weight] : request->cost_weights()) {
    if (agent->SetWeightByName(name, weight) == -1) {
      std::ostringstream error_string;
      error_string << "Weight '" << name
                   << "' not found in task. Available names are:\n";
      auto* agent_model = agent->GetModel();
      for (int i = 0; i < agent_model->nsensor &&
                      agent_model->sensor_type[i] == mjSENS_USER;
           i++) {
        std::string_view sensor_name(agent_model->names +
                                     agent_model->name_sensoradr[i]);
        error_string << "  " << sensor_name << "\n";
      }
      return {grpc::StatusCode::INVALID_ARGUMENT, error_string.str()};
    }
  }

  return grpc::Status::OK;
}

}  // namespace grpc_agent_util
