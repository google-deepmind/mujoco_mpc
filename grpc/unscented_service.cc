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

#include "grpc/unscented_service.h"

#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/log/check.h>
#include <absl/status/status.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>
#include <absl/strings/strip.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "grpc/unscented.pb.h"
#include "mjpc/estimators/unscented.h"

namespace unscented_grpc {

using ::unscented::CovarianceRequest;
using ::unscented::CovarianceResponse;
using ::unscented::InitRequest;
using ::unscented::InitResponse;
using ::unscented::NoiseRequest;
using ::unscented::NoiseResponse;
using ::unscented::ResetRequest;
using ::unscented::ResetResponse;
using ::unscented::SettingsRequest;
using ::unscented::SettingsResponse;
using ::unscented::StateRequest;
using ::unscented::StateResponse;
using ::unscented::TimersRequest;
using ::unscented::TimersResponse;
using ::unscented::UpdateRequest;
using ::unscented::UpdateResponse;

// TODO(taylor): make CheckSize utility function
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

#define CHECK_SIZE(name, n1, n2)                              \
  {                                                           \
    auto expr = (CheckSize(name, n1, n2));                    \
    if (!(expr).ok()) {                                       \
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, \
                          (expr).ToString());                 \
    }                                                         \
  }

UnscentedService::~UnscentedService() {}

grpc::Status UnscentedService::Init(grpc::ServerContext* context,
                                    const unscented::InitRequest* request,
                                    unscented::InitResponse* response) {
  // ----- initialize with model ----- //
  mjpc::UniqueMjModel tmp_model = {nullptr, mj_deleteModel};

  // convert message
  if (request->has_model() && request->model().has_mjb()) {
    std::string_view mjb = request->model().mjb();
    static constexpr char file[] = "temporary-filename.mjb";
    // mjVFS structs need to be allocated on the heap, because it's ~2MB
    auto vfs = std::make_unique<mjVFS>();
    mj_defaultVFS(vfs.get());
    mj_makeEmptyFileVFS(vfs.get(), file, mjb.size());
    int file_idx = mj_findFileVFS(vfs.get(), file);
    memcpy(vfs->filedata[file_idx], mjb.data(), mjb.size());
    tmp_model = {mj_loadModel(file, vfs.get()), mj_deleteModel};
    mj_deleteFileVFS(vfs.get(), file);
  } else if (request->has_model() && request->model().has_xml()) {
    std::string_view model_xml = request->model().xml();
    char load_error[1024] = "";

    // TODO(taylor): utilize grpc_agent_util method
    static constexpr char file[] = "temporary-filename.xml";
    // mjVFS structs need to be allocated on the heap, because it's ~2MB
    auto vfs = std::make_unique<mjVFS>();
    mj_defaultVFS(vfs.get());
    mj_makeEmptyFileVFS(vfs.get(), file, model_xml.size());
    int file_idx = mj_findFileVFS(vfs.get(), file);
    memcpy(vfs->filedata[file_idx], model_xml.data(), model_xml.size());
    tmp_model = {mj_loadXML(file, vfs.get(), load_error, sizeof(load_error)),
                 mj_deleteModel};
    mj_deleteFileVFS(vfs.get(), file);
  } else {
    mju_error("Failed to create mjModel.");
  }

  // move
  model_override_ = std::move(tmp_model);
  mjModel* model = model_override_.get();

  // initialize unscented
  unscented_.Initialize(model);
  unscented_.Reset();

  return grpc::Status::OK;
}

grpc::Status UnscentedService::Reset(grpc::ServerContext* context,
                                     const unscented::ResetRequest* request,
                                     unscented::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  unscented_.Reset();

  return grpc::Status::OK;
}

grpc::Status UnscentedService::Settings(
    grpc::ServerContext* context, const unscented::SettingsRequest* request,
    unscented::SettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings
  unscented::Settings input = request->settings();
  unscented::Settings* output = response->mutable_settings();

  // epsilon
  if (input.has_alpha()) {
    unscented_.settings.alpha = input.alpha();
  }
  output->set_alpha(unscented_.settings.alpha);

  // beta
  if (input.has_beta()) {
    unscented_.settings.beta = input.beta();
  }
  output->set_beta(unscented_.settings.beta);

  return grpc::Status::OK;
}

grpc::Status UnscentedService::Update(
    grpc::ServerContext* context,
    const unscented::UpdateRequest* request,
    unscented::UpdateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // update
  unscented_.Update(request->ctrl().data(),
                               request->sensor().data());

  return grpc::Status::OK;
}

grpc::Status UnscentedService::Timers(grpc::ServerContext* context,
                                      const unscented::TimersRequest* request,
                                      unscented::TimersResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // measurement
  response->set_update(unscented_.TimerUpdate());

  return grpc::Status::OK;
}

grpc::Status UnscentedService::State(grpc::ServerContext* context,
                                     const unscented::StateRequest* request,
                                     unscented::StateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  unscented::State input = request->state();
  unscented::State* output = response->mutable_state();

  // set state
  int nstate = unscented_.model->nq + unscented_.model->nv;
  if (input.state_size() > 0) {
    CHECK_SIZE("state", nstate, input.state_size());
    mju_copy(unscented_.state.data(), input.state().data(), nstate);
  }

  // get state
  double* state = unscented_.state.data();
  for (int i = 0; i < nstate; i++) {
    output->add_state(state[i]);
  }

  return grpc::Status::OK;
}

grpc::Status UnscentedService::Covariance(
    grpc::ServerContext* context, const unscented::CovarianceRequest* request,
    unscented::CovarianceResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  unscented::Covariance input = request->covariance();
  unscented::Covariance* output = response->mutable_covariance();

  // dimensions
  int nvelocity = 2 * unscented_.model->nv;
  int ncovariance = nvelocity * nvelocity;

  // set dimension
  output->set_dimension(nvelocity);

  // set covariance
  if (input.covariance_size() > 0) {
    CHECK_SIZE("covariance", ncovariance, input.covariance_size());
    mju_copy(unscented_.covariance.data(), input.covariance().data(),
             ncovariance);
  }

  // get covariance
  double* covariance = unscented_.covariance.data();
  for (int i = 0; i < ncovariance; i++) {
    output->add_covariance(covariance[i]);
  }

  return grpc::Status::OK;
}

grpc::Status UnscentedService::Noise(grpc::ServerContext* context,
                                     const unscented::NoiseRequest* request,
                                     unscented::NoiseResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  unscented::Noise input = request->noise();
  unscented::Noise* output = response->mutable_noise();

  // dimensions
  int nprocess = 2 * unscented_.model->nv;
  int nsensor = unscented_.model->nsensordata;

  // set process noise
  if (input.process_size() > 0) {
    CHECK_SIZE("process noise", nprocess, input.process_size());
    mju_copy(unscented_.noise_process.data(), input.process().data(), nprocess);
  }

  // get process noise
  double* process = unscented_.noise_process.data();
  for (int i = 0; i < nprocess; i++) {
    output->add_process(process[i]);
  }

  // set sensor noise
  if (input.sensor_size() > 0) {
    CHECK_SIZE("sensor noise", nsensor, input.sensor_size());
    mju_copy(unscented_.noise_sensor.data(), input.sensor().data(), nsensor);
  }

  // get sensor noise
  double* sensor = unscented_.noise_sensor.data();
  for (int i = 0; i < nsensor; i++) {
    output->add_sensor(sensor[i]);
  }

  return grpc::Status::OK;
}

#undef CHECK_SIZE

}  // namespace unscented_grpc
