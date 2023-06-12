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

#include "grpc/estimator_service.h"

#include <string_view>
#include <vector>

#include <absl/log/check.h>
#include <absl/strings/str_format.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>
#include "grpc/estimator.pb.h"
#include "mjpc/estimators/estimator.h"

namespace estimator_grpc {

using ::estimator::InitRequest;
using ::estimator::InitResponse;
using ::estimator::SetDataRequest;
using ::estimator::SetDataResponse;
using ::estimator::GetDataRequest;
using ::estimator::GetDataResponse;
using ::estimator::SetSettingsRequest;
using ::estimator::SetSettingsResponse;
using ::estimator::GetSettingsRequest;
using ::estimator::GetSettingsResponse;
using ::estimator::GetCostsRequest;
using ::estimator::GetCostsResponse;
using ::estimator::SetWeightsRequest;
using ::estimator::SetWeightsResponse;
using ::estimator::GetWeightsRequest;
using ::estimator::GetWeightsResponse;
using ::estimator::ShiftTrajectoriesRequest;
using ::estimator::ShiftTrajectoriesResponse;
using ::estimator::ResetRequest;
using ::estimator::ResetResponse;
using ::estimator::OptimizeRequest;
using ::estimator::OptimizeResponse;
using ::estimator::GetStatusRequest;
using ::estimator::GetStatusResponse;

EstimatorService::~EstimatorService() {}

grpc::Status EstimatorService::Init(
    grpc::ServerContext *context,
    const estimator::InitRequest *request,
    estimator::InitResponse *response)
{
  // ----- initialize with model ----- //
  mjpc::UniqueMjModel tmp_model = {nullptr, mj_deleteModel};

  // convert message
  if (request->has_model() && request->model().has_xml()) {
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
  estimator_model_override_ = std::move(tmp_model);

  // initialize estimator 
  estimator_.Initialize(estimator_model_override_.get());

  // set estimation horizon 
  estimator_.SetConfigurationLength(request->configuration_length());

  return grpc::Status::OK;
}

grpc::Status EstimatorService::SetData(
    grpc::ServerContext *context,
    const estimator::SetDataRequest *request,
    estimator::SetDataResponse *response)
{
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // valid index 
  int index = (int)(request->index());
  if (index < 0 || index >= estimator_.configuration_length_) {
    // TODO(taylor): does this need a warning/error message or StatusCode?
    return grpc::Status::CANCELLED;
  }

  // data
  estimator::Data data = request->data();

  // set configuration
  if (data.configuration_size() == estimator_.model_->nq) {
    estimator_.configuration_.Set(data.configuration().data(), index);
  }

  // set velocity
  if (data.velocity_size() == estimator_.model_->nv) {
    estimator_.velocity_.Set(data.velocity().data(), index);
  }

  // set acceleration
  if (data.acceleration_size() == estimator_.model_->nv) {
    estimator_.acceleration_.Set(data.acceleration().data(), index);
  }

  // set action
  if (data.action_size() == estimator_.model_->nu) {
    estimator_.action_.Set(data.action().data(), index);
  }

  // set time
  if (data.time_size() == 1) {
    estimator_.time_.Set(data.time().data(), index);
  }

  // set configuration prior
  if (data.configuration_prior_size() == estimator_.model_->nq) {
    estimator_.configuration_prior_.Set(data.configuration_prior().data(),
                                        index);
  }

  // set sensor measurement
  if (data.sensor_measurement_size() == estimator_.dim_sensor_) {
    estimator_.sensor_measurement_.Set(data.sensor_measurement().data(),
                                       index);
  }

  // set sensor prediction
  if (data.sensor_prediction_size() == estimator_.dim_sensor_) {
    estimator_.sensor_prediction_.Set(data.sensor_prediction().data(),
                                      index);
  }

  // set force measurement
  if (data.force_measurement_size() == estimator_.model_->nv) {
    estimator_.force_measurement_.Set(data.force_measurement().data(),
                                      index);
  }

  // set force prediction
  if (data.force_prediction_size() == estimator_.model_->nv) {
    estimator_.force_prediction_.Set(data.force_prediction().data(), index);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::GetData(
    grpc::ServerContext* context, const estimator::GetDataRequest* request,
    estimator::GetDataResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // valid index
  int index = (int)(request->index());
  if (index < 0 || index >= estimator_.configuration_length_) {
    // TODO(taylor): does this need a warning/error message or StatusCode?
    return grpc::Status::CANCELLED;
  }

  // data 
  estimator::Data* data = response->mutable_data();

  // get configuration
  double* configuration = estimator_.configuration_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.model_->nq; i++) {
    data->add_configuration(configuration[i]);
  }

  // get velocity
  double* velocity = estimator_.velocity_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.model_->nv; i++) {
    data->add_velocity(velocity[i]);
  }

  // get acceleration
  double* acceleration = estimator_.acceleration_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.model_->nv; i++) {
    data->add_acceleration(acceleration[i]);
  }

  // get action
  double* action = estimator_.action_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.model_->nu; i++) {
    data->add_action(action[i]);
  }

  // get time
  double* time = estimator_.time_.Get(index);

  // copy to response
  data->add_time(time[0]);

  // get configuration prior
  double* configuration_prior = estimator_.configuration_prior_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.model_->nq; i++) {
    data->add_configuration_prior(configuration_prior[i]);
  }

  // get sensor measurement
  double* sensor_measurement = estimator_.sensor_measurement_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.dim_sensor_; i++) {
    data->add_sensor_measurement(sensor_measurement[i]);
  }

  // get sensor prediction
  double* sensor_prediction = estimator_.sensor_prediction_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.dim_sensor_; i++) {
    data->add_sensor_prediction(sensor_prediction[i]);
  }

  // get force measurement
  double* force_measurement = estimator_.force_measurement_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.model_->nv; i++) {
    data->add_force_measurement(force_measurement[i]);
  }

  // get force prediction
  double* force_prediction = estimator_.force_prediction_.Get(index);

  // copy to response
  for (int i = 0; i < estimator_.model_->nv; i++) {
    data->add_force_prediction(force_prediction[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::SetSettings(
    grpc::ServerContext *context,
    const estimator::SetSettingsRequest *request,
    estimator::SetSettingsResponse *response)
{
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings 
  estimator::Settings settings = request->settings();

  // configuration length
  if (settings.has_configuration_length()) {
    // unpack
    int configuration_length = (int)(settings.configuration_length());

    // check for valid length
    if (configuration_length < 3) {
      // TODO(taylor): warning/error?
      return grpc::Status::CANCELLED;
    }

    // set
    estimator_.configuration_length_ = configuration_length;
  }

  // search type 
  if (settings.has_search_type()) {
    // unpack 
    mjpc::SearchType search_type = (mjpc::SearchType)(settings.search_type());

    // check for valid search type 
    if (search_type >= mjpc::kNumSearch) {
      // TODO(taylor): warning/error?
      return grpc::Status::CANCELLED;
    }

    // set 
    estimator_.search_type_ = search_type;
  }

  // prior flag 
  if (settings.has_prior_flag()) 
    estimator_.prior_flag_ = settings.prior_flag();

  // sensor flag 
  if (settings.has_sensor_flag()) 
    estimator_.sensor_flag_ = settings.sensor_flag();

  // force flag 
  if (settings.has_force_flag()) 
    estimator_.force_flag_ = settings.force_flag();

  // smoother iterations 
  if (settings.has_smoother_iterations()) {
    // unpack 
    int iterations = settings.smoother_iterations();

    // test valid 
    if (iterations < 1) {
      // TODO(taylor): warning/error ?
      return grpc::Status::CANCELLED;
    }

    // set
    estimator_.max_smoother_iterations_ = settings.smoother_iterations();
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::GetSettings(
    grpc::ServerContext* context, const estimator::GetSettingsRequest* request,
    estimator::GetSettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings 
  estimator::Settings* settings = response->mutable_settings();

  // configuration length
  settings->set_configuration_length(estimator_.configuration_length_);

  // search type
  settings->set_search_type(estimator_.search_type_);

  // prior flag
  settings->set_prior_flag(estimator_.prior_flag_);

  // sensor flag
  settings->set_sensor_flag(estimator_.sensor_flag_);

  // force flag
  settings->set_force_flag(estimator_.force_flag_);

  // smoother iterations
  settings->set_smoother_iterations(estimator_.max_smoother_iterations_);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::GetCosts(
    grpc::ServerContext* context, const estimator::GetCostsRequest* request,
    estimator::GetCostsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  // costs 
  estimator::Cost* costs = response->mutable_cost();

  // cost 
  costs->set_total(estimator_.cost_);

  // prior cost 
  costs->set_prior(estimator_.cost_prior_);

  // sensor cost 
  costs->set_sensor(estimator_.cost_sensor_);

  // force cost 
  costs->set_force(estimator_.cost_force_);

  // initial cost 
  costs->set_initial(estimator_.cost_initial_);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::SetWeights(
    grpc::ServerContext *context,
    const estimator::SetWeightsRequest *request,
    estimator::SetWeightsResponse *response)
{
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // weight 
  estimator::Weight weight = request->weight();

  // prior 
  if (weight.has_prior()) {
    estimator_.scale_prior_ = weight.prior();
  }

  // sensor
  int ns = estimator_.dim_sensor_;
  if (weight.sensor_size() == ns) {
    mju_copy(estimator_.scale_sensor_.data(), weight.sensor().data(), ns);
  }

  // force
  int nv = estimator_.model_->nv;
  if (weight.force_size() == nv) {
    mju_copy(estimator_.scale_force_.data(), weight.force().data(), nv);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::GetWeights(
    grpc::ServerContext* context, const estimator::GetWeightsRequest* request,
    estimator::GetWeightsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // weight 
  estimator::Weight* weight = response->mutable_weight();

  // prior 
  weight->set_prior(estimator_.scale_prior_);

  // sensor 
  for (int i = 0; i < estimator_.dim_sensor_; i++) {
    weight->add_sensor(estimator_.scale_sensor_[i]);
  }

  // force 
  for (int i = 0; i < estimator_.model_->nv; i++) {
    weight->add_force(estimator_.scale_force_[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::ShiftTrajectories(
    grpc::ServerContext* context, const estimator::ShiftTrajectoriesRequest* request,
    estimator::ShiftTrajectoriesResponse* response) {
  // if (!Initialized()) {
  //   return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  // }

  // shift
  estimator_.ShiftTrajectoryHead(request->shift());

  // get head index
  response->set_head(estimator_.configuration_.head_index_);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Reset(
    grpc::ServerContext* context, const estimator::ResetRequest* request,
    estimator::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  estimator_.Reset();

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Optimize(
    grpc::ServerContext* context, const estimator::OptimizeRequest* request,
    estimator::OptimizeResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // optimize
  estimator_.Optimize(thread_pool_);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::GetStatus(
    grpc::ServerContext* context, const estimator::GetStatusRequest* request,
    estimator::GetStatusResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // status 
  estimator::Status* status = response->mutable_status();

  // search iterations
  status->set_search_iterations(estimator_.iterations_line_search_);

  // smoother iterations
  status->set_smoother_iterations(estimator_.iterations_smoother_);

  // step size
  status->set_step_size(estimator_.step_size_);

  // regularization
  status->set_regularization(estimator_.regularization_);

  // gradient norm
  status->set_gradient_norm(estimator_.gradient_norm_);

  return grpc::Status::OK;
}

}  // namespace estimator_grpc
