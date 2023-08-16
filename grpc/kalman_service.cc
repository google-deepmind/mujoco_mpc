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

#include "grpc/kalman_service.h"

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

#include "grpc/kalman.pb.h"
#include "mjpc/estimators/kalman.h"

namespace kalman_grpc {

using ::kalman::CovarianceRequest;
using ::kalman::CovarianceResponse;
using ::kalman::InitRequest;
using ::kalman::InitResponse;
using ::kalman::NoiseRequest;
using ::kalman::NoiseResponse;
using ::kalman::ResetRequest;
using ::kalman::ResetResponse;
using ::kalman::SettingsRequest;
using ::kalman::SettingsResponse;
using ::kalman::StateRequest;
using ::kalman::StateResponse;
using ::kalman::TimersRequest;
using ::kalman::TimersResponse;
using ::kalman::UpdateMeasurementRequest;
using ::kalman::UpdateMeasurementResponse;
using ::kalman::UpdatePredictionRequest;
using ::kalman::UpdatePredictionResponse;

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

KalmanService::~KalmanService() {}

grpc::Status KalmanService::Init(grpc::ServerContext* context,
                                 const kalman::InitRequest* request,
                                 kalman::InitResponse* response) {
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
  kalman_model_override_ = std::move(tmp_model);
  mjModel* model = kalman_model_override_.get();

  // initialize kalman
  kalman_.Initialize(model);
  kalman_.Reset();

  return grpc::Status::OK;
}

grpc::Status KalmanService::Reset(grpc::ServerContext* context,
                                  const kalman::ResetRequest* request,
                                  kalman::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  kalman_.Reset();

  return grpc::Status::OK;
}

grpc::Status KalmanService::Settings(grpc::ServerContext* context,
                                     const kalman::SettingsRequest* request,
                                     kalman::SettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings
  kalman::Settings input = request->settings();
  kalman::Settings* output = response->mutable_settings();

  // epsilon
  if (input.has_epsilon()) {
    kalman_.settings.epsilon = input.epsilon();
  }
  output->set_epsilon(kalman_.settings.epsilon);

  // flg_centered
  if (input.has_flg_centered()) {
    kalman_.settings.flg_centered = input.flg_centered();
  }
  output->set_flg_centered(kalman_.settings.flg_centered);

  return grpc::Status::OK;
}

grpc::Status KalmanService::UpdateMeasurement(
    grpc::ServerContext* context,
    const kalman::UpdateMeasurementRequest* request,
    kalman::UpdateMeasurementResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // measurement update
  kalman_.UpdateMeasurement(request->ctrl().data(), request->sensor().data());

  return grpc::Status::OK;
}

grpc::Status KalmanService::UpdatePrediction(
    grpc::ServerContext* context,
    const kalman::UpdatePredictionRequest* request,
    kalman::UpdatePredictionResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // prediction update
  kalman_.UpdatePrediction();

  return grpc::Status::OK;
}

grpc::Status KalmanService::Timers(grpc::ServerContext* context,
                                   const kalman::TimersRequest* request,
                                   kalman::TimersResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // measurement
  response->set_measurement(kalman_.TimerMeasurement());

  // prediction
  response->set_prediction(kalman_.TimerPrediction());

  return grpc::Status::OK;
}

grpc::Status KalmanService::State(grpc::ServerContext* context,
                                  const kalman::StateRequest* request,
                                  kalman::StateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  kalman::State input = request->state();
  kalman::State* output = response->mutable_state();

  // set state
  int nstate = kalman_.model->nq + kalman_.model->nv;
  if (input.state_size() > 0) {
    CHECK_SIZE("state", nstate, input.state_size());
    mju_copy(kalman_.state.data(), input.state().data(), nstate);
  }

  // get state
  double* state = kalman_.state.data();
  for (int i = 0; i < nstate; i++) {
    output->add_state(state[i]);
  }

  return grpc::Status::OK;
}

grpc::Status KalmanService::Covariance(grpc::ServerContext* context,
                                       const kalman::CovarianceRequest* request,
                                       kalman::CovarianceResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  kalman::Covariance input = request->covariance();
  kalman::Covariance* output = response->mutable_covariance();

  // dimensions
  int nvelocity = 2 * kalman_.model->nv;
  int ncovariance = nvelocity * nvelocity;

  // set dimension
  output->set_dimension(nvelocity);

  // set covariance
  if (input.covariance_size() > 0) {
    CHECK_SIZE("covariance", ncovariance, input.covariance_size());
    mju_copy(kalman_.covariance.data(), input.covariance().data(), ncovariance);
  }

  // get covariance
  double* covariance = kalman_.covariance.data();
  for (int i = 0; i < ncovariance; i++) {
    output->add_covariance(covariance[i]);
  }

  return grpc::Status::OK;
}

grpc::Status KalmanService::Noise(grpc::ServerContext* context,
                                  const kalman::NoiseRequest* request,
                                  kalman::NoiseResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // unpack input/output
  kalman::Noise input = request->noise();
  kalman::Noise* output = response->mutable_noise();

  // dimensions
  int nprocess = 2 * kalman_.model->nv;
  int nsensor = kalman_.model->nsensordata;

  // set process noise
  if (input.process_size() > 0) {
    CHECK_SIZE("process noise", nprocess, input.process_size());
    mju_copy(kalman_.noise_process.data(), input.process().data(), nprocess);
  }

  // get process noise
  double* process = kalman_.noise_process.data();
  for (int i = 0; i < nprocess; i++) {
    output->add_process(process[i]);
  }

  // set sensor noise
  if (input.sensor_size() > 0) {
    CHECK_SIZE("sensor noise", nsensor, input.sensor_size());
    mju_copy(kalman_.noise_sensor.data(), input.sensor().data(), nsensor);
  }

  // get sensor noise
  double* sensor = kalman_.noise_sensor.data();
  for (int i = 0; i < nsensor; i++) {
    output->add_sensor(sensor[i]);
  }

  return grpc::Status::OK;
}

#undef CHECK_SIZE

}  // namespace kalman_grpc
