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

  // set configuration
  if (request->configuration_size() == estimator_.model_->nq) {
    estimator_.configuration_.Set(request->configuration().data(), index);
  }

  // set velocity
  if (request->velocity_size() == estimator_.model_->nv) {
    estimator_.velocity_.Set(request->velocity().data(), index);
  }

  // set acceleration
  if (request->acceleration_size() == estimator_.model_->nv) {
    estimator_.acceleration_.Set(request->acceleration().data(), index);
  }

  // set action
  if (request->action_size() == estimator_.model_->nu) {
    estimator_.action_.Set(request->action().data(), index);
  }

  // set time
  if (request->time_size() == 1) {
    estimator_.time_.Set(request->time().data(), index);
  }

  // set configuration prior
  if (request->configuration_prior_size() == estimator_.model_->nq) {
    estimator_.configuration_prior_.Set(request->configuration_prior().data(),
                                        index);
  }

  // set sensor measurement
  if (request->sensor_measurement_size() == estimator_.dim_sensor_) {
    estimator_.sensor_measurement_.Set(request->sensor_measurement().data(),
                                       index);
  }

  // set sensor prediction
  if (request->sensor_prediction_size() == estimator_.dim_sensor_) {
    estimator_.sensor_prediction_.Set(request->sensor_prediction().data(),
                                      index);
  }

  // set force measurement
  if (request->force_measurement_size() == estimator_.model_->nv) {
    estimator_.force_measurement_.Set(request->force_measurement().data(),
                                      index);
  }

  // set force prediction
  if (request->force_prediction_size() == estimator_.model_->nv) {
    estimator_.force_prediction_.Set(request->force_prediction().data(), index);
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

  // get configuration
  if (request->has_configuration() && request->configuration() == true) {
    // get data
    double* configuration = estimator_.configuration_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.model_->nq; i++) {
      response->add_configuration(configuration[i]);
    }
  }

  // get velocity
  if (request->has_velocity() && request->velocity() == true) {
    // get data
    double* velocity = estimator_.velocity_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.model_->nv; i++) {
      response->add_velocity(velocity[i]);
    }
  }

  // get acceleration
  if (request->has_acceleration() && request->acceleration() == true) {
    // get data
    double* acceleration = estimator_.acceleration_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.model_->nv; i++) {
      response->add_acceleration(acceleration[i]);
    }
  }

  // get action
  if (request->has_action() && request->action() == true) {
    // get data
    double* action = estimator_.action_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.model_->nu; i++) {
      response->add_action(action[i]);
    }
  }

  // get time
  if (request->has_time() && request->time() == true) {
    // get data
    double* time = estimator_.time_.Get(index);

    // copy to response
    response->add_time(time[0]);
  }

  // get configuration prior
  if (request->has_configuration_prior() &&
      request->configuration_prior() == true) {
    // get data
    double* configuration_prior = estimator_.configuration_prior_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.model_->nq; i++) {
      response->add_configuration_prior(configuration_prior[i]);
    }
  }

  // get sensor measurement
  if (request->has_sensor_measurement() &&
      request->sensor_measurement() == true) {
    // get data
    double* sensor_measurement = estimator_.sensor_measurement_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.dim_sensor_; i++) {
      response->add_sensor_measurement(sensor_measurement[i]);
    }
  }

  // get sensor prediction
  if (request->has_sensor_prediction() &&
      request->sensor_prediction() == true) {
    // get data
    double* sensor_prediction = estimator_.sensor_prediction_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.dim_sensor_; i++) {
      response->add_sensor_prediction(sensor_prediction[i]);
    }
  }

  // get force measurement
  if (request->has_force_measurement() &&
      request->force_measurement() == true) {
    // get data
    double* force_measurement = estimator_.force_measurement_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.model_->nv; i++) {
      response->add_force_measurement(force_measurement[i]);
    }
  }

  // get force prediction
  if (request->has_force_prediction() && request->force_prediction() == true) {
    // get data
    double* force_prediction = estimator_.force_prediction_.Get(index);

    // copy to response
    for (int i = 0; i < estimator_.model_->nv; i++) {
      response->add_force_prediction(force_prediction[i]);
    }
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

  // configuration length
  if (request->has_configuration_length()) {
    // unpack
    int configuration_length = (int)(request->configuration_length());

    // check for valid length
    if (configuration_length < 3) {
      // TODO(taylor): warning/error?
      return grpc::Status::CANCELLED;
    }

    // set
    estimator_.configuration_length_ = configuration_length;
  }

  // search type 
  if (request->has_search_type()) {
    // unpack 
    mjpc::SearchType search_type = (mjpc::SearchType)(request->search_type());

    // check for valid search type 
    if (search_type >= mjpc::kNumSearch) {
      // TODO(taylor): warning/error?
      return grpc::Status::CANCELLED;
    }

    // set 
    estimator_.search_type_ = search_type;
  }

  // prior flag 
  if (request->has_prior_flag()) 
    estimator_.prior_flag_ = request->prior_flag();

  // sensor flag 
  if (request->has_sensor_flag()) 
    estimator_.sensor_flag_ = request->sensor_flag();

  // force flag 
  if (request->has_force_flag()) 
    estimator_.force_flag_ = request->force_flag();

  // smoother iterations 
  if (request->has_smoother_iterations()) {
    // unpack 
    int iterations = request->smoother_iterations();

    // test valid 
    if (iterations < 1) {
      // TODO(taylor): warning/error ?
      return grpc::Status::CANCELLED;
    }

    // set
    estimator_.max_smoother_iterations_ = request->smoother_iterations();
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::GetSettings(
    grpc::ServerContext* context, const estimator::GetSettingsRequest* request,
    estimator::GetSettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // configuration length
  if (request->has_configuration_length() &&
      request->configuration_length() == true) {
    response->set_configuration_length(estimator_.configuration_length_);
  }

  // search type
  if (request->has_search_type() && request->search_type() == true) {
    response->set_search_type(estimator_.search_type_);
  }

  // prior flag
  if (request->has_prior_flag() && request->prior_flag() == true) {
    response->set_prior_flag(estimator_.prior_flag_);
  }

  // sensor flag
  if (request->has_sensor_flag() && request->sensor_flag() == true) {
    response->set_sensor_flag(estimator_.sensor_flag_);
  }

  // force flag
  if (request->has_force_flag() && request->force_flag() == true) {
    response->set_force_flag(estimator_.force_flag_);
  }

  // smoother iterations
  if (request->has_smoother_iterations() &&
      request->smoother_iterations() == true) {
    response->set_smoother_iterations(estimator_.max_smoother_iterations_);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::GetCosts(
    grpc::ServerContext* context, const estimator::GetCostsRequest* request,
    estimator::GetCostsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // cost 
  if (request->has_cost() && request->cost() == true) {
    response->set_cost(estimator_.cost_);
  }

  // prior cost 
  if (request->has_prior() && request->prior() == true) {
    response->set_prior(estimator_.cost_prior_);
  }

  // sensor cost 
  if (request->has_sensor() && request->sensor() == true) {
    response->set_sensor(estimator_.cost_sensor_);
  }

  // force cost 
  if (request->has_force() && request->force() == true) {
    response->set_force(estimator_.cost_force_);
  }

  // initial cost 
  if (request->has_initial() && request->initial() == true) {
    response->set_initial(estimator_.cost_initial_);
  }

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

  // prior 
  if (request->has_prior()) {
    estimator_.scale_prior_ = request->prior();
  }

  // sensor
  int ns = estimator_.dim_sensor_;
  if (request->sensor_size() == ns) {
    mju_copy(estimator_.scale_sensor_.data(), request->sensor().data(), ns);
  }

  // force
  int nv = estimator_.model_->nv;
  if (request->force_size() == nv) {
    mju_copy(estimator_.scale_force_.data(), request->force().data(), nv);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::GetWeights(
    grpc::ServerContext* context, const estimator::GetWeightsRequest* request,
    estimator::GetWeightsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // prior 
  if (request->has_prior() && request->prior() == true) {
    response->set_prior(estimator_.scale_prior_);
  }

  // sensor 
  if (request->has_sensor() && request->sensor() == true) {
    for (int i = 0; i < estimator_.dim_sensor_; i++) {
      response->add_sensor(estimator_.scale_sensor_[i]);
    }
  }

  // force 
  if (request->has_force() && request->force() == true) {
    for (int i = 0; i < estimator_.model_->nv; i++) {
      response->add_force(estimator_.scale_force_[i]);
    }
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

  // search iterations
  if (request->has_search_iterations() &&
      request->search_iterations() == true) {
    response->set_search_iterations(estimator_.iterations_line_search_);
  }

  // smoother iterations
  if (request->has_smoother_iterations() &&
      request->smoother_iterations() == true) {
    response->set_smoother_iterations(estimator_.iterations_smoother_);
  }

  // step size
  if (request->has_step_size() && request->step_size() == true) {
    response->set_step_size(estimator_.step_size_);
  }

  // regularization
  if (request->has_regularization() && request->regularization() == true) {
    response->set_regularization(estimator_.regularization_);
  }

  // gradient norm
  if (request->has_gradient_norm() && request->gradient_norm() == true) {
    response->set_gradient_norm(estimator_.gradient_norm_);
  }

  return grpc::Status::OK;
}

}  // namespace estimator_grpc
