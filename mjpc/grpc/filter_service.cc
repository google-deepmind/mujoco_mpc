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

#include "mjpc/grpc/filter_service.h"

#include <cstring>
#include <memory>
#include <sstream>
#include <string_view>
#include <utility>
#include <vector>

#include <absl/log/check.h>
#include <absl/status/status.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "mjpc/grpc/filter.pb.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/utilities.h"

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

  // initialize filters
  filter_ = mjpc::GetNumberOrDefault(0, model, "estimator");
  for (const auto& filter : filters_) {
    filter->Initialize(model);
    filter->Reset();
  }

  return grpc::Status::OK;
}

grpc::Status FilterService::Reset(grpc::ServerContext* context,
                                  const filter::ResetRequest* request,
                                  filter::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  filters_[filter_]->Reset();

  return grpc::Status::OK;
}

grpc::Status FilterService::Update(grpc::ServerContext* context,
                                   const filter::UpdateRequest* request,
                                   filter::UpdateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // update
  filters_[filter_]->Update(request->ctrl().data(), request->sensor().data(),
                            request->mode());

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

  // active filter
  mjpc::Estimator* active_filter = filters_[filter_].get();

  // model
  mjModel* model = active_filter->Model();

  // state
  double* state = active_filter->State();

  // set state
  int nstate = model->nq + model->nv + model->na;
  if (input.state_size() > 0) {
    CHECK_SIZE("state", nstate, input.state_size());
    mju_copy(state, input.state().data(), nstate);
  }

  // get state
  for (int i = 0; i < nstate; i++) {
    output->add_state(state[i]);
  }

  // set time
  if (input.has_time()) {
    active_filter->Time() = input.time();
  }

  // get time
  output->set_time(active_filter->Time());

  // get qfrc
  double* qfrc = active_filter->Qfrc();
  for (int i = 0; i < model->nv; i++) {
    output->add_qfrc(qfrc[i]);
  }

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

  // active filter
  mjpc::Estimator* active_filter = filters_[filter_].get();

  // dimensions
  int nvelocity = active_filter->DimensionProcess();
  int ncovariance = nvelocity * nvelocity;

  // set dimension
  output->set_dimension(nvelocity);

  // covariance
  double* covariance = active_filter->Covariance();

  // set covariance
  if (input.covariance_size() > 0) {
    CHECK_SIZE("covariance", ncovariance, input.covariance_size());
    mju_copy(covariance, input.covariance().data(), ncovariance);
  }

  // get covariance
  for (int i = 0; i < ncovariance; i++) {
    output->add_covariance(covariance[i]);
  }

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

  // active filter
  mjpc::Estimator* active_filter = filters_[filter_].get();

  // dimensions
  int nprocess = active_filter->DimensionProcess();
  int nsensor = active_filter->DimensionSensor();

  // noise
  double* process_noise = active_filter->ProcessNoise();
  double* sensor_noise = active_filter->SensorNoise();

  // set process noise
  if (input.process_size() > 0) {
    CHECK_SIZE("process noise", nprocess, input.process_size());
    mju_copy(process_noise, input.process().data(), nprocess);
  }

  // get process noise
  for (int i = 0; i < nprocess; i++) {
    output->add_process(process_noise[i]);
  }

  // set sensor noise
  if (input.sensor_size() > 0) {
    CHECK_SIZE("sensor noise", nsensor, input.sensor_size());
    mju_copy(sensor_noise, input.sensor().data(), nsensor);
  }

  // get sensor noise
  for (int i = 0; i < nsensor; i++) {
    output->add_sensor(sensor_noise[i]);
  }

  return grpc::Status::OK;
}

#undef CHECK_SIZE

}  // namespace filter_grpc
