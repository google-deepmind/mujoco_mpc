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

#include <cstring>
#include <string_view>
#include <vector>

#include "grpc/estimator_service.h"

#include <absl/log/check.h>
#include <absl/strings/str_format.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "grpc/estimator.pb.h"
#include "mjpc/estimators/estimator.h"

namespace estimator_grpc {

using ::estimator::CostHessianRequest;
using ::estimator::CostHessianResponse;
using ::estimator::CostRequest;
using ::estimator::CostResponse;
using ::estimator::DataRequest;
using ::estimator::DataResponse;
using ::estimator::InitRequest;
using ::estimator::InitResponse;
using ::estimator::NormRequest;
using ::estimator::NormResponse;
using ::estimator::OptimizeRequest;
using ::estimator::OptimizeResponse;
using ::estimator::PriorMatrixRequest;
using ::estimator::PriorMatrixResponse;
using ::estimator::ResetBufferRequest;
using ::estimator::ResetBufferResponse;
using ::estimator::ResetRequest;
using ::estimator::ResetResponse;
using ::estimator::SettingsRequest;
using ::estimator::SettingsResponse;
using ::estimator::ShiftRequest;
using ::estimator::ShiftResponse;
using ::estimator::StatusRequest;
using ::estimator::StatusResponse;
using ::estimator::UpdateBufferRequest;
using ::estimator::UpdateBufferResponse;
using ::estimator::WeightsRequest;
using ::estimator::WeightsResponse;

EstimatorService::~EstimatorService() {}

grpc::Status EstimatorService::Init(grpc::ServerContext* context,
                                    const estimator::InitRequest* request,
                                    estimator::InitResponse* response) {
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
  mjModel* model = estimator_model_override_.get();

  // initialize estimator
  estimator_.Initialize(model);

  // set estimation horizon
  estimator_.SetConfigurationLength(request->configuration_length());

  // initialize buffer
  buffer_.Initialize(estimator_.dim_sensor_, estimator_.num_sensor_, model->nu,
                     (request->has_buffer_length() ? request->buffer_length()
                                                   : mjpc::MAX_TRAJECTORY));

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Data(grpc::ServerContext* context,
                                    const estimator::DataRequest* request,
                                    estimator::DataResponse* response) {
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
  estimator::Data input = request->data();
  estimator::Data* output = response->mutable_data();

  // set configuration
  int nq = estimator_.model_->nq;
  if (input.configuration_size() == nq) {
    estimator_.configuration_.Set(input.configuration().data(), index);
  }

  // get configuration
  double* configuration = estimator_.configuration_.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration(configuration[i]);
  }

  // set velocity
  int nv = estimator_.model_->nv;
  if (input.velocity_size() == nv) {
    estimator_.velocity_.Set(input.velocity().data(), index);
  }

  // get velocity
  double* velocity = estimator_.velocity_.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_velocity(velocity[i]);
  }

  // set acceleration
  if (input.acceleration_size() == nv) {
    estimator_.acceleration_.Set(input.acceleration().data(), index);
  }

  // get acceleration
  double* acceleration = estimator_.acceleration_.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_acceleration(acceleration[i]);
  }

  // set time
  if (input.time_size() == 1) {
    estimator_.time_.Set(input.time().data(), index);
  }

  // get time
  double* time = estimator_.time_.Get(index);
  output->add_time(time[0]);

  // set configuration prior
  if (input.configuration_prior_size() == nq) {
    estimator_.configuration_prior_.Set(input.configuration_prior().data(),
                                        index);
  }

  // get configuration prior
  double* prior = estimator_.configuration_prior_.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration_prior(prior[i]);
  }

  // set sensor measurement
  int ns = estimator_.dim_sensor_;
  if (input.sensor_measurement_size() == ns) {
    estimator_.sensor_measurement_.Set(input.sensor_measurement().data(),
                                       index);
  }

  // get sensor measurement
  double* sensor_measurement = estimator_.sensor_measurement_.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_measurement(sensor_measurement[i]);
  }

  // set sensor prediction
  if (input.sensor_prediction_size() == ns) {
    estimator_.sensor_prediction_.Set(input.sensor_prediction().data(), index);
  }

  // get sensor prediction
  double* sensor_prediction = estimator_.sensor_prediction_.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_prediction(sensor_prediction[i]);
  }

  // set force measurement
  if (input.force_measurement_size() == nv) {
    estimator_.force_measurement_.Set(input.force_measurement().data(), index);
  }

  // get force measurement
  double* force_measurement = estimator_.force_measurement_.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_measurement(force_measurement[i]);
  }

  // set force prediction
  if (input.force_prediction_size() == nv) {
    estimator_.force_prediction_.Set(input.force_prediction().data(), index);
  }

  // get force prediction
  double* force_prediction = estimator_.force_prediction_.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_prediction(force_prediction[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Settings(
    grpc::ServerContext* context, const estimator::SettingsRequest* request,
    estimator::SettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings
  estimator::Settings input = request->settings();
  estimator::Settings* output = response->mutable_settings();

  // configuration length
  if (input.has_configuration_length()) {
    // unpack
    int configuration_length = (int)(input.configuration_length());

    // check for valid length
    if (configuration_length < 3) {
      // TODO(taylor): warning/error?
      return grpc::Status::CANCELLED;
    }

    // set
    estimator_.configuration_length_ = configuration_length;
  }
  output->set_configuration_length(estimator_.configuration_length_);

  // search type
  if (input.has_search_type()) {
    // unpack
    mjpc::SearchType search_type = (mjpc::SearchType)(input.search_type());

    // check for valid search type
    if (search_type >= mjpc::kNumSearch) {
      // TODO(taylor): warning/error?
      return grpc::Status::CANCELLED;
    }

    // set
    estimator_.search_type_ = search_type;
  }
  output->set_search_type(estimator_.search_type_);

  // prior flag
  if (input.has_prior_flag()) estimator_.prior_flag_ = input.prior_flag();
  output->set_prior_flag(estimator_.prior_flag_);

  // sensor flag
  if (input.has_sensor_flag()) estimator_.sensor_flag_ = input.sensor_flag();
  output->set_sensor_flag(estimator_.sensor_flag_);

  // force flag
  if (input.has_force_flag()) estimator_.force_flag_ = input.force_flag();
  output->set_force_flag(estimator_.force_flag_);

  // smoother iterations
  if (input.has_smoother_iterations()) {
    // unpack
    int iterations = input.smoother_iterations();

    // test valid
    if (iterations < 1) {
      // TODO(taylor): warning/error ?
      return grpc::Status::CANCELLED;
    }

    // set
    estimator_.max_smoother_iterations_ = input.smoother_iterations();
  }
  output->set_smoother_iterations(estimator_.max_smoother_iterations_);

  // skip prior weight update
  if (input.has_skip_prior_weight_update()) {
    estimator_.skip_update_prior_weight = input.skip_prior_weight_update();
  }
  output->set_skip_prior_weight_update(estimator_.skip_update_prior_weight);

  // time scaling
  if (input.has_time_scaling()) {
    estimator_.time_scaling_ = input.time_scaling();
  }
  output->set_time_scaling(estimator_.time_scaling_);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Cost(grpc::ServerContext* context,
                                    const estimator::CostRequest* request,
                                    estimator::CostResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  // costs
  estimator::Cost* cost = response->mutable_cost();

  // cost
  cost->set_total(estimator_.cost_);

  // prior cost
  cost->set_prior(estimator_.cost_prior_);

  // sensor cost
  cost->set_sensor(estimator_.cost_sensor_);

  // force cost
  cost->set_force(estimator_.cost_force_);

  // initial cost
  cost->set_initial(estimator_.cost_initial_);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Weights(grpc::ServerContext* context,
                                       const estimator::WeightsRequest* request,
                                       estimator::WeightsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // weight
  estimator::Weight input = request->weight();
  estimator::Weight* output = response->mutable_weight();

  // prior
  if (input.has_prior()) {
    estimator_.scale_prior_ = input.prior();
  }
  output->set_prior(estimator_.scale_prior_);

  // sensor
  int num_sensor = estimator_.num_sensor_;
  if (input.sensor_size() == num_sensor) {
    mju_copy(estimator_.scale_sensor_.data(), input.sensor().data(),
             num_sensor);
  }
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor(estimator_.scale_sensor_[i]);
  }

  // force
  int num_jnt = 4;
  if (input.force_size() == num_jnt) {
    mju_copy(estimator_.scale_force_.data(), input.force().data(), num_jnt);
  }
  for (int i = 0; i < num_jnt; i++) {
    output->add_force(estimator_.scale_force_[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Norms(grpc::ServerContext* context,
                                     const estimator::NormRequest* request,
                                     estimator::NormResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // norm
  estimator::Norm input = request->norm();
  estimator::Norm* output = response->mutable_norm();

  // set sensor type
  int num_sensor = estimator_.num_sensor_;
  if (input.sensor_type_size() == num_sensor) {
    std::memcpy(estimator_.norm_sensor_.data(), input.sensor_type().data(),
                num_sensor * sizeof(int));
  }

  // get sensor type
  mjpc::NormType* sensor_type = estimator_.norm_sensor_.data();
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor_type((int)sensor_type[i]);
  }

  // set sensor parameters
  if (input.sensor_parameters_size() == num_sensor * 3) {
    mju_copy(estimator_.norm_parameters_sensor_.data(),
             input.sensor_parameters().data(), num_sensor * 3);
  }

  // get sensor parameters 
  double* sensor_parameters = estimator_.norm_parameters_sensor_.data();
  for (int i = 0; i < 3 * num_sensor; i++) {
    output->add_sensor_parameters(sensor_parameters[i]);
  }

  // set force type
  int nj = 4;
  if (input.force_type_size() == nj) {
    std::memcpy(estimator_.norm_force_, input.force_type().data(),
                nj * sizeof(int));
  }

  // get force type
  mjpc::NormType* force_type = estimator_.norm_force_;
  for (int i = 0; i < nj; i++) {
    output->add_force_type((int)force_type[i]);
  }

  // set force parameters
  int nfp = 12;
  if (input.sensor_parameters_size() == nfp) {
    mju_copy(estimator_.norm_parameters_force_[0],
             input.force_parameters().data() + 0, 3);
    mju_copy(estimator_.norm_parameters_force_[1],
             input.force_parameters().data() + 3, 3);
    mju_copy(estimator_.norm_parameters_force_[2],
             input.force_parameters().data() + 6, 3);
    mju_copy(estimator_.norm_parameters_force_[3],
             input.force_parameters().data() + 9, 3);
  }

  // get force parameters
  double* fp0 = estimator_.norm_parameters_force_[0];
  for (int i = 0; i < 3; i++) {
    output->add_force_parameters(fp0[i]);
  }
  double* fp1 = estimator_.norm_parameters_force_[1];
  for (int i = 0; i < 3; i++) {
    output->add_force_parameters(fp1[i]);
  }
  double* fp2 = estimator_.norm_parameters_force_[2];
  for (int i = 0; i < 3; i++) {
    output->add_force_parameters(fp2[i]);
  }
  double* fp3 = estimator_.norm_parameters_force_[3];
  for (int i = 0; i < 3; i++) {
    output->add_force_parameters(fp3[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Shift(grpc::ServerContext* context,
                                     const estimator::ShiftRequest* request,
                                     estimator::ShiftResponse* response) {
  // if (!Initialized()) {
  //   return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  // }

  // shift
  estimator_.ShiftTrajectoryHead(request->shift());

  // get head index
  response->set_head(estimator_.configuration_.head_index_);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Reset(grpc::ServerContext* context,
                                     const estimator::ResetRequest* request,
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

grpc::Status EstimatorService::Status(grpc::ServerContext* context,
                                      const estimator::StatusRequest* request,
                                      estimator::StatusResponse* response) {
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

grpc::Status EstimatorService::CostHessian(
    grpc::ServerContext* context, const estimator::CostHessianRequest* request,
    estimator::CostHessianResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // dimension
  int dim = estimator_.model_->nv * estimator_.configuration_length_;
  response->set_dimension(dim);

  // get cost Hessian
  // TODO(taylor): return only upper triangle?
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      double data = estimator_.cost_hessian_[dim * i + j];
      response->add_hessian(data);
    }
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::PriorMatrix(
    grpc::ServerContext* context, const estimator::PriorMatrixRequest* request,
    estimator::PriorMatrixResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // dimension
  int dim = estimator_.model_->nv * estimator_.configuration_length_;
  response->set_dimension(dim);

  // set prior matrix
  // TODO(taylor): loop over upper triangle only
  if (request->prior_size() == dim * dim) {
    mju_copy(estimator_.weight_prior_dense_.data(), request->prior().data(),
             dim * dim);
  }

  // get prior matrix
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      response->add_prior(estimator_.weight_prior_dense_[dim * i + j]);
    }
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::ResetBuffer(
    grpc::ServerContext* context, const estimator::ResetBufferRequest* request,
    estimator::ResetBufferResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  buffer_.Reset();

  return grpc::Status::OK;
}

grpc::Status EstimatorService::BufferData(
    grpc::ServerContext* context, const estimator::BufferDataRequest* request,
    estimator::BufferDataResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // valid index
  int index = request->index();
  if (index < 0 || index >= buffer_.Length()) {
    // TODO(taylor): does this need a warning/error message or StatusCode?
    return grpc::Status::CANCELLED;
  }

  // buffer
  estimator::Buffer input = request->buffer();
  estimator::Buffer* output = response->mutable_buffer();

  // set sensor
  int ns = estimator_.dim_sensor_;
  if (input.sensor_size() == ns) {
    buffer_.sensor_.Set(input.sensor().data(), index);
  }

  // get sensor
  double* sensor = buffer_.sensor_.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor(sensor[i]);
  }

  // set mask
  int num_sensor = estimator_.num_sensor_;
  if (input.mask_size() == num_sensor) {
    buffer_.sensor_mask_.Set(input.mask().data(), index);
  }

  // get mask
  int* mask = buffer_.sensor_mask_.Get(index);
  for (int i = 0; i < num_sensor; i++) {
    output->add_mask(mask[i]);
  }

  // set ctrl
  int nu = estimator_.model_->nu;
  if (input.ctrl_size() == nu) {
    buffer_.ctrl_.Set(input.ctrl().data(), index);
  }

  // get ctrl
  double* ctrl = buffer_.ctrl_.Get(index);
  for (int i = 0; i < nu; i++) {
    output->add_ctrl(ctrl[i]);
  }

  // set time
  if (input.time_size() == 1) {
    buffer_.time_.Set(input.time().data(), index);
  }

  // get time
  double* time = buffer_.time_.Get(index);
  output->add_time(time[0]);

  // get length
  response->set_length(buffer_.Length());

  return grpc::Status::OK;
}

grpc::Status EstimatorService::UpdateBuffer(
    grpc::ServerContext* context, const estimator::UpdateBufferRequest* request,
    estimator::UpdateBufferResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // buffer
  estimator::Buffer buffer = request->buffer();

  // check for all data
  if (buffer.sensor_size() != estimator_.dim_sensor_)
    return {grpc::StatusCode::FAILED_PRECONDITION, "Missing sensor."};
  if (buffer.mask_size() != estimator_.num_sensor_)
    return {grpc::StatusCode::FAILED_PRECONDITION, "Missing sensor mask."};
  if (buffer.ctrl_size() != estimator_.model_->nu)
    return {grpc::StatusCode::FAILED_PRECONDITION, "Missing ctrl."};
  if (buffer.time_size() != 1)
    return {grpc::StatusCode::FAILED_PRECONDITION, "Missing time."};

  // update
  buffer_.Update(buffer.sensor().data(), buffer.mask().data(),
                 buffer.ctrl().data(), buffer.time().data()[0]);

  // get length
  response->set_length(buffer_.Length());

  return grpc::Status::OK;
}

}  // namespace estimator_grpc
