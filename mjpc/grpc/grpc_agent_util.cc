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

#include "mjpc/grpc/grpc_agent_util.h"

#include <memory>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/log/check.h>
#include <absl/status/status.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>
#include <absl/strings/strip.h>
#include <grpcpp/support/status.h>
#include <mujoco/mjdata.h>
#include <mujoco/mjmodel.h>
#include <mujoco/mujoco.h>

#include "mjpc/grpc/agent.pb.h"
#include "mjpc/agent.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"

namespace grpc_agent_util {

using ::agent::GetActionRequest;
using ::agent::GetActionResponse;
using ::agent::GetCostValuesAndWeightsRequest;
using ::agent::GetCostValuesAndWeightsResponse;
using ::agent::GetModeRequest;
using ::agent::GetModeResponse;
using ::agent::GetStateResponse;
using ::agent::GetTaskParametersRequest;
using ::agent::GetTaskParametersResponse;
using ::agent::SetCostWeightsRequest;
using ::agent::SetModeRequest;
using ::agent::SetStateRequest;
using ::agent::SetTaskParametersRequest;
using ::agent::ValueAndWeight;

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

namespace {
// TODO(nimrod): make planner a const reference
std::vector<double> AverageAction(mjpc::Planner& planner, const mjModel* model,
                                  bool nominal_action, mjData* rollout_data,
                                  mjpc::State* rollout_state, double time,
                                  double averaging_duration) {
  int nu = model->nu;
  std::vector<double> ret(nu, 0);
  int nactions = 0;
  double end_time = time + averaging_duration;

  if (nominal_action) {
    std::vector<double> action(nu, 0);
    while (time < end_time) {
      planner.ActionFromPolicy(action.data(), /*state=*/nullptr, time);
      mju_addTo(ret.data(), action.data(), nu);
      time += model->opt.timestep;
      nactions++;
    }
  } else {
    rollout_data->time = time;
    while (rollout_data->time <= end_time) {
      rollout_state->Set(model, rollout_data);
      const double* state = rollout_state->state().data();
      planner.ActionFromPolicy(rollout_data->ctrl, state,
                                              rollout_data->time);
      mju_addTo(ret.data(), rollout_data->ctrl, nu);
      mj_step(model, rollout_data);
      nactions++;
    }
  }
  mju_scl(ret.data(), ret.data(), 1.0 / nactions, nu);
  return ret;
}

}  // namespace
grpc::Status GetAction(const GetActionRequest* request,
                       const mjpc::Agent* agent,
                       const mjModel* model, mjData* rollout_data,
                       mjpc::State* rollout_state,
                       GetActionResponse* response) {
  double time =
      request->has_time() ? request->time() : agent->state.time();

  if (request->averaging_duration() > 0) {
    if (request->nominal_action()) {
      rollout_data = nullptr;
      rollout_state = nullptr;
    } else {
      agent->state.CopyTo(model, rollout_data);
      rollout_state->Set(model, rollout_data);
    }
    std::vector<double> ret = AverageAction(agent->ActivePlanner(), model,
                        request->nominal_action(), rollout_data, rollout_state,
                        time, request->averaging_duration());
    response->mutable_action()->Assign(ret.begin(), ret.end());
  } else {
    std::vector<double> ret(model->nu, 0);
    const double* state = request->nominal_action()
                              ? nullptr
                              : agent->state.state().data();
    agent->ActivePlanner().ActionFromPolicy(ret.data(), state, time);
    response->mutable_action()->Assign(ret.begin(), ret.end());
  }

  return grpc::Status::OK;
}

grpc::Status GetCostValuesAndWeights(
    const GetCostValuesAndWeightsRequest* request, const mjpc::Agent* agent,
    const mjModel* model, mjData* data,
    GetCostValuesAndWeightsResponse* response) {
  const mjModel* agent_model = agent->GetModel();
  const mjpc::Task* task = agent->ActiveTask();
  std::vector<double> residuals(task->num_residual, 0);  // scratch space
  double terms[mjpc::kMaxCostTerms];
  task->Residual(model, data, residuals.data());
  task->UnweightedCostTerms(terms, residuals.data());
  for (int i = 0; i < task->num_term; i++) {
    CHECK_EQ(agent_model->sensor_type[i], mjSENS_USER);
    std::string_view sensor_name(agent_model->names +
                                  agent_model->name_sensoradr[i]);
    ValueAndWeight value_and_weight;
    value_and_weight.set_value(terms[i]);
    value_and_weight.set_weight(task->weight[i]);
    (*response->mutable_values_weights())[sensor_name] = value_and_weight;
  }
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
  agent->ActiveTask()->UpdateResidual();

  return grpc::Status::OK;
}

grpc::Status GetTaskParameters(const GetTaskParametersRequest* request,
                               mjpc::Agent* agent,
                               GetTaskParametersResponse* response) {
  mjModel* agent_model = agent->GetModel();
  int shift = 0;
  for (int i = 0; i < agent_model->nnumeric; i++) {
    std::string_view numeric_name(agent_model->names +
                                  agent_model->name_numericadr[i]);
    if (absl::StartsWith(numeric_name, "residual_select_")) {
      std::string_view name =
          absl::StripPrefix(numeric_name, "residual_select_");
      (*response->mutable_parameters())[name].set_selection(
          mjpc::ResidualSelection(agent_model, name,
                                  agent->ActiveTask()->parameters[shift]));
      shift++;
    } else if (absl::StartsWith(numeric_name, "residual_")) {
      std::string_view name = absl::StripPrefix(numeric_name, "residual_");
      (*response->mutable_parameters())[name].set_numeric(
          agent->ActiveTask()->parameters[shift]);
      shift++;
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

grpc::Status SetMode(const SetModeRequest* request, mjpc::Agent* agent) {
  int outcome = agent->SetModeByName(request->mode());
  if (outcome == -1) {
    std::vector<std::string> mode_names = agent->GetAllModeNames();
    std::ostringstream error_string;
    error_string << "Mode '" << request->mode()
                  << "' not found in task. Available names are:\n";
    for (const auto& mode_name : mode_names) {
      error_string << "  " << mode_name << "\n";
    }
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error_string.str());
  } else {
    return grpc::Status::OK;
  }
}

grpc::Status GetMode(const GetModeRequest* request, mjpc::Agent* agent,
                     GetModeResponse* response) {
  response->set_mode(agent->GetModeName());
  return grpc::Status::OK;
}

mjpc::UniqueMjModel LoadModelFromString(std::string_view xml, char* error,
                             int error_size) {
  static constexpr char file[] = "temporary-filename.xml";
  // mjVFS structs need to be allocated on the heap, because it's ~2MB
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_makeEmptyFileVFS(vfs.get(), file, xml.size());
  int file_idx = mj_findFileVFS(vfs.get(), file);
  memcpy(vfs->filedata[file_idx], xml.data(), xml.size());
  mjpc::UniqueMjModel m = {mj_loadXML(file, vfs.get(), error, error_size),
                           mj_deleteModel};
  mj_deleteFileVFS(vfs.get(), file);
  return m;
}

mjpc::UniqueMjModel LoadModelFromBytes(std::string_view mjb) {
  static constexpr char file[] = "temporary-filename.mjb";
  // mjVFS structs need to be allocated on the heap, because it's ~2MB
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_makeEmptyFileVFS(vfs.get(), file, mjb.size());
  int file_idx = mj_findFileVFS(vfs.get(), file);
  memcpy(vfs->filedata[file_idx], mjb.data(), mjb.size());
  mjpc::UniqueMjModel m = {mj_loadModel(file, vfs.get()), mj_deleteModel};
  mj_deleteFileVFS(vfs.get(), file);
  return m;
}

grpc::Status InitAgent(mjpc::Agent* agent, const agent::InitRequest* request) {
  std::string_view task_id = request->task_id();
  int task_index = agent->GetTaskIdByName(task_id);
  if (task_index == -1) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        absl::StrFormat("Invalid task_id: '%s'", task_id));
  }
  agent->gui_task_id = task_index;

  mjpc::UniqueMjModel tmp_model = {nullptr, mj_deleteModel};
  char load_error[1024] = "";

  if (request->has_model() && request->model().has_mjb()) {
    std::string_view model_mjb_bytes = request->model().mjb();
    // TODO(khartikainen): Add error handling for mjb loading.
    tmp_model = LoadModelFromBytes(model_mjb_bytes);
  } else if (request->has_model() && request->model().has_xml()) {
    std::string_view model_xml = request->model().xml();
    tmp_model = LoadModelFromString(model_xml, load_error, sizeof(load_error));
  }
  agent->OverrideModel(std::move(tmp_model));
  return grpc::Status::OK;
}
}  // namespace grpc_agent_util
