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

#include <cstring>
#include <string_view>
#include <vector>

#include <absl/log/check.h>
#include <absl/status/status.h>
#include <absl/strings/match.h>
#include <absl/strings/str_format.h>
#include <absl/strings/strip.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "grpc/estimator.pb.h"
#include "mjpc/estimators/estimator.h"

namespace estimator_grpc {

using ::estimator::CostGradientRequest;
using ::estimator::CostGradientResponse;
using ::estimator::CostHessianRequest;
using ::estimator::CostHessianResponse;
using ::estimator::CostRequest;
using ::estimator::CostResponse;
using ::estimator::DataRequest;
using ::estimator::DataResponse;
using ::estimator::InitRequest;
using ::estimator::InitResponse;
using ::estimator::InitializeDataRequest;
using ::estimator::InitializeDataResponse;
using ::estimator::InitialStateRequest;
using ::estimator::InitialStateResponse;
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
using ::estimator::TimingRequest;
using ::estimator::TimingResponse;
using ::estimator::UpdateRequest;
using ::estimator::UpdateResponse;
using ::estimator::UpdateBufferRequest;
using ::estimator::UpdateBufferResponse;
using ::estimator::UpdateDataRequest;
using ::estimator::UpdateDataResponse;
using ::estimator::WeightsRequest;
using ::estimator::WeightsResponse;

// TODO(taylor): make CheckSize utility function for agent and estimator
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

#define CHECK_SIZE(name, n1, n2)                            \
{                                                           \
  auto expr = (CheckSize(name, n1, n2));                    \
  if (!(expr).ok()) {                                       \
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, \
                        (expr).ToString());                 \
  }                                                         \
}

EstimatorService::~EstimatorService() {}

grpc::Status EstimatorService::Init(grpc::ServerContext* context,
                                    const estimator::InitRequest* request,
                                    estimator::InitResponse* response) {
  // check configuration length
  if (request->configuration_length() < mjpc::MIN_HISTORY ||
      request->configuration_length() > mjpc::MAX_HISTORY) {
    return {grpc::StatusCode::OUT_OF_RANGE, "Invalid configuration length."};
  }

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
    return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
  }

  // data
  estimator::Data input = request->data();
  estimator::Data* output = response->mutable_data();

  // set configuration
  int nq = estimator_.model_->nq;
  if (input.configuration_size() > 0) {
    CHECK_SIZE("configuration", nq, input.configuration_size());
    estimator_.configuration.Set(input.configuration().data(), index);
  }

  // get configuration
  double* configuration = estimator_.configuration.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration(configuration[i]);
  }

  // set velocity
  int nv = estimator_.model_->nv;
  if (input.velocity_size() > 0) {
    CHECK_SIZE("velocity", nv, input.velocity_size());
    estimator_.velocity.Set(input.velocity().data(), index);
  }

  // get velocity
  double* velocity = estimator_.velocity.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_velocity(velocity[i]);
  }

  // set acceleration
  if (input.acceleration_size() > 0) {
    CHECK_SIZE("acceleration", nv, input.acceleration_size());
    estimator_.acceleration.Set(input.acceleration().data(), index);
  }

  // get acceleration
  double* acceleration = estimator_.acceleration.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_acceleration(acceleration[i]);
  }

  // set time
  if (input.time_size() > 0) {
    CHECK_SIZE("time", 1, input.time_size());
    estimator_.time.Set(input.time().data(), index);
  }

  // get time
  double* time = estimator_.time.Get(index);
  output->add_time(time[0]);

  // set ctrl 
  int nu = estimator_.model_->nu;
  if (input.ctrl_size() > 0) {
    CHECK_SIZE("ctrl", nu, input.ctrl_size());
    estimator_.ctrl.Set(input.ctrl().data(), index);
  }

  // get ctrl 
  double* ctrl = estimator_.ctrl.Get(index);
  for (int i = 0; i < nu; i++) {
    output->add_ctrl(ctrl[i]);
  }

  // set previous configuration
  if (input.configuration_prior_size() > 0) {
    CHECK_SIZE("configuration_previous", nq, input.configuration_prior_size());
    estimator_.configuration_previous.Set(input.configuration_previous().data(),
                                        index);
  }

  // get configuration previous
  double* prior = estimator_.configuration_previous.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration_prior(prior[i]);
  }

  // set sensor measurement
  int ns = estimator_.dim_sensor_;
  if (input.sensor_measurement_size() > 0) {
    CHECK_SIZE("sensor_measurement", ns, input.sensor_measurement_size());
    estimator_.sensor_measurement.Set(input.sensor_measurement().data(),
                                       index);
  }

  // get sensor measurement
  double* sensor_measurement = estimator_.sensor_measurement.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_measurement(sensor_measurement[i]);
  }

  // set sensor prediction
  if (input.sensor_prediction_size() > 0) {
    CHECK_SIZE("sensor_prediction", ns, input.sensor_prediction_size());
    estimator_.sensor_prediction.Set(input.sensor_prediction().data(), index);
  }

  // get sensor prediction
  double* sensor_prediction = estimator_.sensor_prediction.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_prediction(sensor_prediction[i]);
  }

  // set sensor mask 
  int num_sensor = estimator_.num_sensor_;
  if (input.sensor_mask_size() > 0) {
    CHECK_SIZE("sensor_mask", num_sensor, input.sensor_mask_size());
    estimator_.sensor_mask.Set(input.sensor_mask().data(), index);
  }

  // get sensor mask 
  int* sensor_mask = estimator_.sensor_mask.Get(index);
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor_mask(sensor_mask[i]);
  }

  // set force measurement
  if (input.force_measurement_size() > 0) {
    CHECK_SIZE("force_measurement", nv, input.force_measurement_size());
    estimator_.force_measurement.Set(input.force_measurement().data(), index);
  }

  // get force measurement
  double* force_measurement = estimator_.force_measurement.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_measurement(force_measurement[i]);
  }

  // set force prediction
  if (input.force_prediction_size() > 0) {
    CHECK_SIZE("force_prediction", nv, input.force_prediction_size());
    estimator_.force_prediction.Set(input.force_prediction().data(), index);
  }

  // get force prediction
  double* force_prediction = estimator_.force_prediction.Get(index);
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
    if (configuration_length < mjpc::MIN_HISTORY ||
        configuration_length > mjpc::MAX_HISTORY) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid configuration length."};
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
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
    }

    // set
    estimator_.search_type = search_type;
  }
  output->set_search_type(estimator_.search_type);

  // prior flag
  if (input.has_prior_flag()) estimator_.prior_flag = input.prior_flag();
  output->set_prior_flag(estimator_.prior_flag);

  // sensor flag
  if (input.has_sensor_flag()) estimator_.sensor_flag = input.sensor_flag();
  output->set_sensor_flag(estimator_.sensor_flag);

  // force flag
  if (input.has_force_flag()) estimator_.force_flag = input.force_flag();
  output->set_force_flag(estimator_.force_flag);

  // smoother iterations
  if (input.has_smoother_iterations()) {
    // unpack
    int iterations = input.smoother_iterations();

    // test valid
    if (iterations < 1) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid smoother iterations."};
    }

    // set
    estimator_.max_smoother_iterations = input.smoother_iterations();
  }
  output->set_smoother_iterations(estimator_.max_smoother_iterations);

  // skip prior weight update
  if (input.has_skip_prior_weight_update()) {
    estimator_.skip_update_prior_weight = input.skip_prior_weight_update();
  }
  output->set_skip_prior_weight_update(estimator_.skip_update_prior_weight);

  // time scaling
  if (input.has_time_scaling()) {
    estimator_.time_scaling = input.time_scaling();
  }
  output->set_time_scaling(estimator_.time_scaling);

  // update prior weight 
  if (input.has_update_prior_weight()) {
    estimator_.update_prior_weight = input.update_prior_weight();
  }
  output->set_update_prior_weight(estimator_.update_prior_weight);

  // regularization initialization 
  if (input.has_regularization_initial()) {
    estimator_.regularization_initial = input.regularization_initial();
  }
  output->set_regularization_initial(estimator_.regularization_initial);

  // gradient tolerance 
  if (input.has_gradient_tolerance()) {
    estimator_.gradient_tolerance = input.gradient_tolerance();
  }
  output->set_gradient_tolerance(estimator_.gradient_tolerance);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Cost(grpc::ServerContext* context,
                                    const estimator::CostRequest* request,
                                    estimator::CostResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // evaluate cost 
  estimator_.cost = estimator_.Cost(thread_pool_);

  // costs
  estimator::Cost* cost = response->mutable_cost();

  // cost
  cost->set_total(estimator_.cost);

  // prior cost
  cost->set_prior(estimator_.cost_prior);

  // sensor cost
  cost->set_sensor(estimator_.cost_sensor);

  // force cost
  cost->set_force(estimator_.cost_force);

  // initial cost
  cost->set_initial(estimator_.cost_initial);

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
    estimator_.scale_prior = input.prior();
  }
  output->set_prior(estimator_.scale_prior);

  // sensor
  int num_sensor = estimator_.num_sensor_;
  if (input.sensor_size() > 0) {
    CHECK_SIZE("scale_sensor", num_sensor, input.sensor_size());
    estimator_.scale_sensor.assign(input.sensor().begin(),
                                    input.sensor().end());
  }
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor(estimator_.scale_sensor[i]);
  }

  // force
  if (input.force_size() > 0) {
    CHECK_SIZE("scale_force", mjpc::NUM_FORCE_TERMS, input.force_size());
    estimator_.scale_force.assign(input.force().begin(), input.force().end());
  }
  for (int i = 0; i < mjpc::NUM_FORCE_TERMS; i++) {
    output->add_force(estimator_.scale_force[i]);
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
  if (input.sensor_type_size() > 0) {
    CHECK_SIZE("sensor_type", num_sensor, input.sensor_type_size());
    estimator_.norm_sensor.clear();
    estimator_.norm_sensor.reserve(num_sensor);
    for (const auto& sensor_type : input.sensor_type()) {
      estimator_.norm_sensor.push_back(
          static_cast<mjpc::NormType>(sensor_type));
    }
  }

  // get sensor type
  for (const auto& sensor_type : estimator_.norm_sensor) {
    output->add_sensor_type(sensor_type);
  }

  // set sensor parameters
  if (input.sensor_parameters_size() > 0) {
    CHECK_SIZE("sensor_parameters", mjpc::MAX_NORM_PARAMETERS * num_sensor,
               input.sensor_parameters_size());
    estimator_.norm_parameters_sensor.assign(input.sensor_parameters().begin(),
                                              input.sensor_parameters().end());
  }

  // get sensor parameters
  for (const auto& sensor_parameters : estimator_.norm_parameters_sensor) {
    output->add_sensor_parameters(sensor_parameters);
  }

  // set force type
  if (input.force_type_size() > 0) {
    CHECK_SIZE("force_type", mjpc::NUM_FORCE_TERMS, input.force_type_size());
    for (int i = 0; i < mjpc::NUM_FORCE_TERMS; i++) {
      estimator_.norm_force[i] =
          static_cast<mjpc::NormType>(input.force_type(i));
    }
  }

  // get force type
  mjpc::NormType* force_type = estimator_.norm_force;
  for (int i = 0; i < mjpc::NUM_FORCE_TERMS; i++) {
    output->add_force_type(force_type[i]);
  }

  // set force parameters
  int nfp = mjpc::NUM_FORCE_TERMS * mjpc::MAX_NORM_PARAMETERS;
  if (input.force_parameters_size() > 0) {
    CHECK_SIZE("force_parameters", nfp, input.force_parameters_size());
    estimator_.norm_parameters_force.assign(
        input.force_parameters().begin(), input.force_parameters().end());
  }

  // get force parameters
  for (int i = 0; i < nfp; i++) {
    output->add_force_parameters(estimator_.norm_parameters_force[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::Shift(grpc::ServerContext* context,
                                     const estimator::ShiftRequest* request,
                                     estimator::ShiftResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // shift
  estimator_.Shift(request->shift());

  // get head index
  response->set_head(estimator_.configuration.Head());

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

grpc::Status EstimatorService::InitializeData(
    grpc::ServerContext* context, const estimator::InitializeDataRequest* request,
    estimator::InitializeDataResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // initialize trajectories with buffer data
  estimator_.InitializeTrajectories(buffer_.sensor_, buffer_.sensor_mask,
                                    buffer_.ctrl, buffer_.time);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::UpdateData(
    grpc::ServerContext* context, const estimator::UpdateDataRequest* request,
    estimator::UpdateDataResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // check for valid number of new elements
  int num_new = request->num_new();
  if (num_new < 1 || num_new > buffer_.Length()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "num_new invalid."};
  }

  // update trajectories with num_new most recent elements from buffer
  estimator_.UpdateTrajectories_(num_new, buffer_.sensor_, buffer_.sensor_mask,
                                 buffer_.ctrl, buffer_.time);

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

grpc::Status EstimatorService::Update(grpc::ServerContext* context,
                                      const estimator::UpdateRequest* request,
                                      estimator::UpdateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // update
  int num_new = estimator_.Update(buffer_, thread_pool_);

  // set num new
  response->set_num_new(num_new);

  return grpc::Status::OK;
}

grpc::Status EstimatorService::InitialState(grpc::ServerContext* context,
                                      const estimator::InitialStateRequest* request,
                                      estimator::InitialStateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // input output
  estimator::State input = request->state();
  estimator::State* output = response->mutable_state();

  // set qpos
  int nq = estimator_.model_->nq;
  if (input.qpos_size() > 0) {
    CHECK_SIZE("qpos", nq, input.qpos_size());
    estimator_.qpos0_.assign(input.qpos().begin(), input.qpos().end());
  }

  // get qpos 
  for (int i = 0; i < nq; i++) {
    output->add_qpos(estimator_.qpos0_[i]);
  }

  // qvel
  int nv = estimator_.model_->nv;
  if (input.qvel_size() > 0) {
    CHECK_SIZE("qvel", nv, input.qvel_size());
    estimator_.qvel0_.assign(input.qvel().begin(), input.qvel().end());
  }

  // get qvel
  for (int i = 0; i < nv; i++) {
    output->add_qvel(estimator_.qvel0_[i]);
  }

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

grpc::Status EstimatorService::Timing(grpc::ServerContext* context,
                                      const estimator::TimingRequest* request,
                                      estimator::TimingResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // double timer_total = 1;
  // double timer_inverse_dynamics_derivatives = 2;
  // double timer_velacc_derivatives = 3;
  // double timer_jacobian_prior = 4;
  // double timer_jacobian_sensor = 5;
  // double timer_jacobian_force = 6;
  // double timer_jacobian_total = 7;
  // double timer_cost_prior_derivatives = 8;
  // double timer_cost_sensor_derivatives = 9;
  // double timer_cost_force_derivatives = 10;
  // double timer_cost_total_derivatives = 11;
  // double timer_cost_gradient = 12;
  // double timer_cost_hessian = 13;
  // double timer_cost_derivatives = 14;
  // double timer_cost = 15;
  // double timer_cost_prior = 16;
  // double timer_cost_sensor = 17;
  // double timer_cost_force = 18;
  // double timer_cost_config_to_velacc = 19;
  // double timer_cost_prediction = 20;
  // double timer_residual_prior = 21;
  // double timer_residual_sensor = 22;
  // double timer_residual_force = 23;
  // double timer_search_direction = 24;
  // double timer_search = 25;
  // double timer_configuration_update = 26;
  // double timer_optimize = 27;
  // double timer_prior_weight_update = 28;
  // double timer_prior_set_weight = 29;
  // double timer_update_trajectory = 30;

  return grpc::Status::OK;
}

grpc::Status EstimatorService::TotalGradient(
    grpc::ServerContext* context, const estimator::CostGradientRequest* request,
    estimator::CostGradientResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // dimension
  int dim = estimator_.model_->nv * estimator_.configuration_length_;

  // get cost gradient
  for (int i = 0; i < dim; i++) {
    response->add_gradient(estimator_.cost_gradient[i]);
  }

  return grpc::Status::OK;
}

grpc::Status EstimatorService::TotalHessian(
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
      double data = estimator_.cost_hessian[dim * i + j];
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
  if (request->prior_size() > 0) {
    CHECK_SIZE("prior_matrix", dim * dim, request->prior_size());
    estimator_.weight_prior.assign(request->prior().begin(),
                                          request->prior().end());
  }

  // get prior matrix
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      response->add_prior(estimator_.weight_prior[dim * i + j]);
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
    return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
  }

  // buffer
  estimator::Buffer input = request->buffer();
  estimator::Buffer* output = response->mutable_buffer();

  // set sensor
  int ns = estimator_.dim_sensor_;
  if (input.sensor_size() > 0) {
    CHECK_SIZE("sensor", ns, input.sensor_size());
    buffer_.sensor_.Set(input.sensor().data(), index);
  }

  // get sensor
  double* sensor = buffer_.sensor_.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor(sensor[i]);
  }

  // set mask
  int num_sensor = estimator_.num_sensor_;
  if (input.mask_size() > 0) {
    CHECK_SIZE("mask", num_sensor, input.mask_size());
    buffer_.sensor_mask.Set(input.mask().data(), index);
  }

  // get mask
  int* mask = buffer_.sensor_mask.Get(index);
  for (int i = 0; i < num_sensor; i++) {
    output->add_mask(mask[i]);
  }

  // set ctrl
  int nu = estimator_.model_->nu;
  if (input.ctrl_size() > 0) {
    CHECK_SIZE("ctrl", nu, input.ctrl_size());
    buffer_.ctrl.Set(input.ctrl().data(), index);
  }

  // get ctrl
  double* ctrl = buffer_.ctrl.Get(index);
  for (int i = 0; i < nu; i++) {
    output->add_ctrl(ctrl[i]);
  }

  // set time
  if (input.time_size() > 0) {
    CHECK_SIZE("time", 1, input.time_size());
    buffer_.time.Set(input.time().data(), index);
  }

  // get time
  double* time = buffer_.time.Get(index);
  output->add_time(time[0]);

  // get length
  response->set_length(buffer_.Length());

  return grpc::Status::OK;
}

#undef CHECK_SIZE

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
