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

#include "grpc/filter_service.h"

#include <absl/log/check.h>
#include <absl/status/status.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>
#include <absl/strings/strip.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include <cstring>
#include <string_view>
#include <utility>
#include <vector>

#include "grpc/filter.pb.h"
#include "mjpc/estimators/include.h"

namespace filter_grpc {

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

FilterService::~FilterService() {}

grpc::Status FilterService::Init(grpc::ServerContext* context,
                                 const filter::InitRequest* request,
                                 filter::InitResponse* response) {
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

  // initialize filter
  filter_.Initialize(model);
  filter_.Reset();

  return grpc::Status::OK;
}

grpc::Status FilterService::Reset(grpc::ServerContext* context,
                                  const filter::ResetRequest* request,
                                  filter::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  filter_.Reset();

  return grpc::Status::OK;
}

grpc::Status FilterService::Settings(grpc::ServerContext* context,
                                     const filter::SettingsRequest* request,
                                     filter::SettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings
  filter::Settings input = request->settings();
  filter::Settings* output = response->mutable_settings();

  // TODO(taylor)

  return grpc::Status::OK;
}

grpc::Status FilterService::Update(grpc::ServerContext* context,
                                   const filter::UpdateRequest* request,
                                   filter::UpdateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // measurement update
  filter_.Update(request->ctrl().data(), request->sensor().data());

  return grpc::Status::OK;
}

grpc::Status FilterService::Timers(grpc::ServerContext* context,
                                   const filter::TimersRequest* request,
                                   filter::TimersResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // TODO(taylor)

  return grpc::Status::OK;
}

grpc::Status FilterService::State(grpc::ServerContext* context,
                                  const filter::StateRequest* request,
                                  filter::StateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  filter::State input = request->state();
  filter::State* output = response->mutable_state();

  // // set state
  // int nstate = filter_.model->nq + filter_.model->nv;
  // if (input.state_size() > 0) {
  //   CHECK_SIZE("state", nstate, input.state_size());
  //   mju_copy(filter_.state.data(), input.state().data(), nstate);
  // }

  // // get state
  // double* state = filter_.state.data();
  // for (int i = 0; i < nstate; i++) {
  //   output->add_state(state[i]);
  // }

  return grpc::Status::OK;
}

grpc::Status FilterService::Covariance(grpc::ServerContext* context,
                                       const filter::CovarianceRequest* request,
                                       filter::CovarianceResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  filter::Covariance input = request->covariance();
  filter::Covariance* output = response->mutable_covariance();

  // dimensions
  // int nvelocity = 2 * filter_.model->nv;
  // int ncovariance = nvelocity * nvelocity;

  // // set dimension
  // output->set_dimension(nvelocity);

  // // set covariance
  // if (input.covariance_size() > 0) {
  //   CHECK_SIZE("covariance", ncovariance, input.covariance_size());
  //   mju_copy(filter_.covariance.data(), input.covariance().data(),
  //   ncovariance);
  // }

  // // get covariance
  // double* covariance = filter_.covariance.data();
  // for (int i = 0; i < ncovariance; i++) {
  //   output->add_covariance(covariance[i]);
  // }

  return grpc::Status::OK;
}

grpc::Status FilterService::Noise(grpc::ServerContext* context,
                                  const filter::NoiseRequest* request,
                                  filter::NoiseResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  filter::Noise input = request->noise();
  filter::Noise* output = response->mutable_noise();

  // // dimensions
  // int nprocess = 2 * filter_.model->nv;
  // int nsensor = filter_.model->nsensordata;

  // // set process noise
  // if (input.process_size() > 0) {
  //   CHECK_SIZE("process noise", nprocess, input.process_size());
  //   mju_copy(filter_.noise_process.data(), input.process().data(), nprocess);
  // }

  // // get process noise
  // double* process = filter_.noise_process.data();
  // for (int i = 0; i < nprocess; i++) {
  //   output->add_process(process[i]);
  // }

  // // set sensor noise
  // if (input.sensor_size() > 0) {
  //   CHECK_SIZE("sensor noise", nsensor, input.sensor_size());
  //   mju_copy(filter_.noise_sensor.data(), input.sensor().data(), nsensor);
  // }

  // // get sensor noise
  // double* sensor = filter_.noise_sensor.data();
  // for (int i = 0; i < nsensor; i++) {
  //   output->add_sensor(sensor[i]);
  // }

  return grpc::Status::OK;
}

#undef CHECK_SIZE

}  // namespace filter_grpc
