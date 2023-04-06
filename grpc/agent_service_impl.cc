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

#include "grpc/agent_service_impl.h"

#include <cstring>
#include <memory>
#include <sstream>
#include <string_view>
#include <vector>

#include <absl/status/status.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>
#include <absl/strings/strip.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>
#include "grpc/agent.pb.h"
#include "mjpc/task.h"
#include "mjpc/tasks/tasks.h"

namespace agent_grpc {

using ::agent::GetActionRequest;
using ::agent::GetActionResponse;
using ::agent::InitRequest;
using ::agent::InitResponse;
using ::agent::PlannerStepRequest;
using ::agent::PlannerStepResponse;
using ::agent::ResetRequest;
using ::agent::ResetResponse;
using ::agent::SetStateRequest;
using ::agent::SetStateResponse;
using ::agent::SetTaskParameterRequest;
using ::agent::SetTaskParameterResponse;

mjpc::Task* task = nullptr;
mjModel* model = nullptr;

void residual_sensor_callback(const mjModel* m, mjData* d, int stage) {
  // with the `m == model` guard in place, no need to clear the callback.
  if (m == model) {
    if (stage == mjSTAGE_ACC) {
      task->Residual(m, d, d->sensordata);
    }
  }
}

mjModel* LoadModelFromString(std::string_view xml, char* error,
                             int error_size) {
  static constexpr char file[] = "temporary-filename.xml";
  // mjVFS structs need to be allocated on the heap, because it's ~2MB
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_makeEmptyFileVFS(vfs.get(), file, xml.size());
  int file_idx = mj_findFileVFS(vfs.get(), file);
  memcpy(vfs->filedata[file_idx], xml.data(), xml.size());
  mjModel* m = mj_loadXML(file, vfs.get(), error, error_size);
  mj_deleteFileVFS(vfs.get(), file);
  return m;
}

mjModel* LoadModelFromBytes(std::string_view mjb) {
  static constexpr char file[] = "temporary-filename.mjb";
  // mjVFS structs need to be allocated on the heap, because it's ~2MB
  auto vfs = std::make_unique<mjVFS>();
  mj_defaultVFS(vfs.get());
  mj_makeEmptyFileVFS(vfs.get(), file, mjb.size());
  int file_idx = mj_findFileVFS(vfs.get(), file);
  memcpy(vfs->filedata[file_idx], mjb.data(), mjb.size());
  mjModel* m = mj_loadModel(file, vfs.get());
  mj_deleteFileVFS(vfs.get(), file);
  return m;
}

grpc::Status AgentServiceImpl::Init(grpc::ServerContext* context,
                                    const InitRequest* request,
                                    InitResponse* response) {
  std::string_view task_id = request->task_id();
  agent_.SetTaskList(mjpc::GetTasks());
  int task_index = agent_.GetTaskIdByName(task_id);
  if (task_index == -1) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        absl::StrFormat("Invalid task_id: '%s'", task_id));
  }

  agent_.SetTaskByIndex(task_index);
  // TODO(khartikainen): is this needed?
  agent_.gui_task_id = task_index;

  mjModel* tmp_model;
  char load_error[1024] = "";

  if (request->has_model() && request->model().has_mjb()) {
    std::string_view model_mjb_bytes = request->model().mjb();
    // TODO(khartikainen): Add error handling for mjb loading.
    tmp_model = LoadModelFromBytes(model_mjb_bytes);
  } else if (request->has_model() && request->model().has_xml()) {
    std::string_view model_xml = request->model().xml();
    tmp_model = LoadModelFromString(model_xml, load_error, sizeof(load_error));
  } else {
    tmp_model = mj_loadXML(agent_.ActiveTask()->XmlPath().c_str(), nullptr,
                           load_error, sizeof(load_error));
  }

  if (!tmp_model) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        absl::StrFormat("Model load error: '%s'", load_error));
  }

  agent_.Initialize(tmp_model);
  mj_deleteModel(tmp_model);
  agent_.Allocate();
  agent_.Reset();

  task = agent_.ActiveTask();
  model = agent_.GetModel();
  data_ = mj_makeData(model);
  mjcb_sensor = residual_sensor_callback;

  agent_.SetState(data_);

  agent_.plan_enabled = true;
  agent_.action_enabled = true;

  return grpc::Status::OK;
}

AgentServiceImpl::~AgentServiceImpl() {
  if (data_) mj_deleteData(data_);
  // no need to delete model and task, since they're owned by agent_.
  model = nullptr;
  task = nullptr;
  mjcb_sensor = nullptr;
}

absl::Status CheckSize(std::string_view name, int model_size, int vector_size) {
  std::ostringstream error_string;
  if (model_size != vector_size) {
    error_string << "expected " << name << " size " << model_size << ", got "
                 << vector_size;
    return absl::InvalidArgumentError(error_string.str());
  }
  return absl::OkStatus();
}

#define CHECK_SIZE(name, n1, n2) \
{ \
  auto expr = (CheckSize(name, n1, n2)); \
if (!(expr).ok()) { \
  return grpc::Status( \
    grpc::StatusCode::INVALID_ARGUMENT, \
    (expr).ToString()); \
  } \
} \

grpc::Status AgentServiceImpl::SetState(grpc::ServerContext* context,
                                        const SetStateRequest* request,
                                        SetStateResponse* response) {
  agent::State state = request->state();

  if (state.has_time()) data_->time = state.time();

  if (0 < state.qpos_size()) {
    CHECK_SIZE("qpos", model->nq, state.qpos_size());
    mju_copy(data_->qpos, state.qpos().data(), model->nq);
  }

  if (0 < state.qvel_size()) {
    CHECK_SIZE("qvel", model->nv, state.qvel_size());
    mju_copy(data_->qvel, state.qvel().data(), model->nv);
  }

  if (0 < state.act_size()) {
    CHECK_SIZE("act", model->na, state.act_size());
    mju_copy(data_->act, state.act().data(), model->na);
  }

  if (0 < state.mocap_pos_size()) {
    CHECK_SIZE("mocap_pos", model->nmocap * 3, state.mocap_pos_size());
    mju_copy(data_->mocap_pos, state.mocap_pos().data(), model->nmocap * 3);
  }

  if (0 < state.mocap_quat_size()) {
    CHECK_SIZE("mocap_quat", model->nmocap * 4, state.mocap_quat_size());
    mju_copy(data_->mocap_quat, state.mocap_quat().data(), model->nmocap * 4);
  }

  if (0 < state.userdata_size()) {
    CHECK_SIZE("userdata", model->nuserdata, state.userdata_size());
    mju_copy(data_->userdata, state.userdata().data(), model->nuserdata);
  }

  agent_.SetState(data_);

  mj_forward(model, data_);
  task->Transition(model, data_);

  return grpc::Status::OK;
}

#undef CHECK_SIZE

grpc::Status AgentServiceImpl::GetAction(grpc::ServerContext* context,
                                         const GetActionRequest* request,
                                         GetActionResponse* response) {
  int nu = agent_.GetActionDim();
  std::vector<double> ret = std::vector<double>(nu);

  double time =
      request->has_time() ? request->time() : agent_.ActiveState().time();

  agent_.ActivePlanner().ActionFromPolicy(
      ret.data(), &agent_.ActiveState().state()[0], time);

  response->mutable_action()->Assign(ret.begin(), ret.end());

  return grpc::Status::OK;
}

grpc::Status AgentServiceImpl::PlannerStep(grpc::ServerContext* context,
                                           const PlannerStepRequest* request,
                                           PlannerStepResponse* response) {
  agent_.plan_enabled = true;
  agent_.PlanIteration(&thread_pool_);

  return grpc::Status::OK;
}

grpc::Status AgentServiceImpl::Reset(grpc::ServerContext* context,
                                     const ResetRequest* request,
                                     ResetResponse* response) {
  agent_.Reset();
  return grpc::Status::OK;
}

grpc::Status AgentServiceImpl::SetTaskParameter(
    grpc::ServerContext* context, const SetTaskParameterRequest* request,
    SetTaskParameterResponse* response) {
  std::string_view name = request->name();
  double value = request->value();

  if (agent_.SetParamByName(name, value) == -1) {
    std::ostringstream error_string;
    error_string << "Parameter " << name
                 << " not found in task.  Available names are:\n";
    auto agent_model = agent_.GetModel();
    for (int i = 0; i < agent_model->nnumeric; i++) {
      std::string_view numeric_name(agent_model->names +
                                    agent_model->name_numericadr[i]);
      if (absl::StartsWith(numeric_name, "residual_")) {
        error_string << absl::StripPrefix(numeric_name, "residual_") << "\n";
      }
    }
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, error_string.str());
  }

  return grpc::Status::OK;
}

}  // namespace agent_grpc
