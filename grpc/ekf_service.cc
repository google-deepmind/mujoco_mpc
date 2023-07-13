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

#include "grpc/ekf_service.h"

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
#include <vector>

#include "grpc/ekf.pb.h"
#include "mjpc/estimators/ekf.h"

namespace ekf_grpc {

using ::ekf::CovarianceRequest;
using ::ekf::CovarianceResponse;
using ::ekf::InitRequest;
using ::ekf::InitResponse;
using ::ekf::NoiseRequest;
using ::ekf::NoiseResponse;
using ::ekf::ResetRequest;
using ::ekf::ResetResponse;
using ::ekf::SettingsRequest;
using ::ekf::SettingsResponse;
using ::ekf::StateRequest;
using ::ekf::StateResponse;
using ::ekf::TimersRequest;
using ::ekf::TimersResponse;
using ::ekf::UpdateMeasurementRequest;
using ::ekf::UpdateMeasurementResponse;
using ::ekf::UpdatePredictionRequest;
using ::ekf::UpdatePredictionResponse;

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

EKFService::~EKFService() {}

grpc::Status EKFService::Init(grpc::ServerContext* context,
                              const ekf::InitRequest* request,
                              ekf::InitResponse* response) {

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
  ekf_model_override_ = std::move(tmp_model);
  mjModel* model = ekf_model_override_.get();

  // initialize ekf
  ekf_.Initialize(model);
  // ekf_.Reset();

  return grpc::Status::OK;
}

grpc::Status EKFService::Reset(grpc::ServerContext* context,
                               const ekf::ResetRequest* request,
                               ekf::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  ekf_.Reset();

  return grpc::Status::OK;
}

grpc::Status EKFService::Settings(grpc::ServerContext* context,
                                  const ekf::SettingsRequest* request,
                                  ekf::SettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings
  ekf::Settings input = request->settings();
  ekf::Settings* output = response->mutable_settings();

  // epsilon 
  if (input.has_epsilon()) {
    ekf_.settings.epsilon = input.epsilon();
  }
  output->set_epsilon(ekf_.settings.epsilon);

  // flg_centered 
  if (input.has_flg_centered()) {
    ekf_.settings.flg_centered = input.flg_centered();
  }
  output->set_flg_centered(ekf_.settings.flg_centered);

  // auto_timestep 
  if (input.has_auto_timestep()) {
    ekf_.settings.auto_timestep = input.auto_timestep();
  }
  output->set_auto_timestep(ekf_.settings.auto_timestep);

  return grpc::Status::OK;
}

grpc::Status EKFService::UpdateMeasurement(
    grpc::ServerContext* context, const ekf::UpdateMeasurementRequest* request,
    ekf::UpdateMeasurementResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // measurement update
  ekf_.UpdateMeasurement(request->ctrl().data(), request->sensor().data());

  return grpc::Status::OK;
}

grpc::Status EKFService::UpdatePrediction(
    grpc::ServerContext* context, const ekf::UpdatePredictionRequest* request,
    ekf::UpdatePredictionResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // prediction update 
  ekf_.UpdatePrediction();

  return grpc::Status::OK;
}

grpc::Status EKFService::Timers(grpc::ServerContext* context,
                                const ekf::TimersRequest* request,
                                ekf::TimersResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // measurement 
  response->set_measurement(ekf_.TimerMeasurement());

  // prediction
  response->set_prediction(ekf_.TimerPrediction());

  return grpc::Status::OK;
}

grpc::Status EKFService::State(grpc::ServerContext* context,
                               const ekf::StateRequest* request,
                               ekf::StateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  ekf::State input = request->state();
  ekf::State* output = response->mutable_state();

  // set state
  int nstate = ekf_.model->nq + ekf_.model->nv;
  if (input.state_size() > 0) {
    CHECK_SIZE("state", nstate, input.state_size());
    mju_copy(ekf_.state.data(), input.state().data(), nstate);
  }

  // get state
  double* state = ekf_.state.data();
  for (int i = 0; i < nstate; i++) {
    output->add_state(state[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EKFService::Covariance(grpc::ServerContext* context,
                                    const ekf::CovarianceRequest* request,
                                    ekf::CovarianceResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  ekf::Covariance input = request->covariance();
  ekf::Covariance* output = response->mutable_covariance();

  // dimensions
  int nvelocity = 2 * ekf_.model->nv;
  int ncovariance = nvelocity * nvelocity;

  // set dimension 
  output->set_dimension(nvelocity);

  // set covariance
  if (input.covariance_size() > 0) {
    CHECK_SIZE("covariance", ncovariance, input.covariance_size());
    mju_copy(ekf_.covariance.data(), input.covariance().data(), ncovariance);
  }

  // get covariance
  double* covariance = ekf_.covariance.data();
  for (int i = 0; i < ncovariance; i++) {
    output->add_covariance(covariance[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EKFService::Noise(grpc::ServerContext* context,
                               const ekf::NoiseRequest* request,
                               ekf::NoiseResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  ekf::Noise input = request->noise();
  ekf::Noise* output = response->mutable_noise();

  // dimensions
  int nprocess = 2 * ekf_.model->nv;
  int nsensor = ekf_.model->nsensordata;

  // set process noise
  if (input.process_size() > 0) {
    CHECK_SIZE("process noise", nprocess, input.process_size());
    mju_copy(ekf_.noise_process.data(), input.process().data(), nprocess);
  }

  // get process noise
  double* process = ekf_.noise_process.data();
  for (int i = 0; i < nprocess; i++) {
    output->add_process(process[i]);
  }

  // set sensor noise
  if (input.sensor_size() > 0) {
    CHECK_SIZE("sensor noise", nsensor, input.sensor_size());
    mju_copy(ekf_.noise_sensor.data(), input.sensor().data(), nsensor);
  }

  // get sensor noise
  double* sensor = ekf_.noise_sensor.data();
  for (int i = 0; i < nsensor; i++) {
    output->add_sensor(sensor[i]);
  }

  return grpc::Status::OK;
}

#undef CHECK_SIZE

}  // namespace ekf_grpc
