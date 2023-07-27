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

#include "grpc/batch_estimator_service.h"

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

#include "grpc/batch_estimator.pb.h"
#include "mjpc/estimators/batch.h"

namespace batch_estimator_grpc {

using ::batch_estimator::CostRequest;
using ::batch_estimator::CostResponse;
using ::batch_estimator::DataRequest;
using ::batch_estimator::DataResponse;
using ::batch_estimator::InitRequest;
using ::batch_estimator::InitResponse;
using ::batch_estimator::NormRequest;
using ::batch_estimator::NormResponse;
using ::batch_estimator::OptimizeRequest;
using ::batch_estimator::OptimizeResponse;
using ::batch_estimator::PriorMatrixRequest;
using ::batch_estimator::PriorMatrixResponse;
using ::batch_estimator::ResetRequest;
using ::batch_estimator::ResetResponse;
using ::batch_estimator::SettingsRequest;
using ::batch_estimator::SettingsResponse;
using ::batch_estimator::ShiftRequest;
using ::batch_estimator::ShiftResponse;
using ::batch_estimator::StatusRequest;
using ::batch_estimator::StatusResponse;
using ::batch_estimator::TimingRequest;
using ::batch_estimator::TimingResponse;
using ::batch_estimator::WeightsRequest;
using ::batch_estimator::WeightsResponse;

// TODO(taylor): make CheckSize utility function for agent and batch_estimator
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

BatchEstimatorService::~BatchEstimatorService() {}

grpc::Status BatchEstimatorService::Init(grpc::ServerContext* context,
                                    const batch_estimator::InitRequest* request,
                                    batch_estimator::InitResponse* response) {
  // check configuration length
  if (request->configuration_length() < mjpc::MIN_HISTORY) {
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

  // move model
  batch_estimator_model_override_ = std::move(tmp_model);

  // initialize batch_estimator
  int length = request->configuration_length();
  batch_estimator_.max_history = length;
  batch_estimator_.Initialize(batch_estimator_model_override_.get());
  batch_estimator_.SetConfigurationLength(length);
  batch_estimator_.Reset();

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Data(grpc::ServerContext* context,
                                    const batch_estimator::DataRequest* request,
                                    batch_estimator::DataResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // valid index
  int index = (int)(request->index());
  if (index < 0 || index >= batch_estimator_.ConfigurationLength()) {
    return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
  }

  // data
  batch_estimator::Data input = request->data();
  batch_estimator::Data* output = response->mutable_data();

  // set configuration
  int nq = batch_estimator_.model->nq;
  if (input.configuration_size() > 0) {
    CHECK_SIZE("configuration", nq, input.configuration_size());
    batch_estimator_.configuration.Set(input.configuration().data(), index);
  }

  // get configuration
  double* configuration = batch_estimator_.configuration.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration(configuration[i]);
  }

  // set velocity
  int nv = batch_estimator_.model->nv;
  if (input.velocity_size() > 0) {
    CHECK_SIZE("velocity", nv, input.velocity_size());
    batch_estimator_.velocity.Set(input.velocity().data(), index);
  }

  // get velocity
  double* velocity = batch_estimator_.velocity.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_velocity(velocity[i]);
  }

  // set acceleration
  if (input.acceleration_size() > 0) {
    CHECK_SIZE("acceleration", nv, input.acceleration_size());
    batch_estimator_.acceleration.Set(input.acceleration().data(), index);
  }

  // get acceleration
  double* acceleration = batch_estimator_.acceleration.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_acceleration(acceleration[i]);
  }

  // set time
  if (input.time_size() > 0) {
    CHECK_SIZE("time", 1, input.time_size());
    batch_estimator_.times.Set(input.time().data(), index);
  }

  // get time
  double* time = batch_estimator_.times.Get(index);
  output->add_time(time[0]);

  // set ctrl
  int nu = batch_estimator_.model->nu;
  if (input.ctrl_size() > 0) {
    CHECK_SIZE("ctrl", nu, input.ctrl_size());
    batch_estimator_.ctrl.Set(input.ctrl().data(), index);
  }

  // get ctrl
  double* ctrl = batch_estimator_.ctrl.Get(index);
  for (int i = 0; i < nu; i++) {
    output->add_ctrl(ctrl[i]);
  }

  // set previous configuration
  if (input.configuration_previous_size() > 0) {
    CHECK_SIZE("configuration_previous", nq,
               input.configuration_previous_size());
    batch_estimator_.configuration_previous.Set(input.configuration_previous().data(),
                                          index);
  }

  // get configuration previous
  double* prior = batch_estimator_.configuration_previous.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration_previous(prior[i]);
  }

  // set sensor measurement
  int ns = batch_estimator_.SensorDimension();
  if (input.sensor_measurement_size() > 0) {
    CHECK_SIZE("sensor_measurement", ns, input.sensor_measurement_size());
    batch_estimator_.sensor_measurement.Set(input.sensor_measurement().data(), index);
  }

  // get sensor measurement
  double* sensor_measurement = batch_estimator_.sensor_measurement.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_measurement(sensor_measurement[i]);
  }

  // set sensor prediction
  if (input.sensor_prediction_size() > 0) {
    CHECK_SIZE("sensor_prediction", ns, input.sensor_prediction_size());
    batch_estimator_.sensor_prediction.Set(input.sensor_prediction().data(), index);
  }

  // get sensor prediction
  double* sensor_prediction = batch_estimator_.sensor_prediction.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_prediction(sensor_prediction[i]);
  }

  // set sensor mask
  int num_sensor = batch_estimator_.NumberSensors();
  if (input.sensor_mask_size() > 0) {
    CHECK_SIZE("sensor_mask", num_sensor, input.sensor_mask_size());
    batch_estimator_.sensor_mask.Set(input.sensor_mask().data(), index);
  }

  // get sensor mask
  int* sensor_mask = batch_estimator_.sensor_mask.Get(index);
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor_mask(sensor_mask[i]);
  }

  // set force measurement
  if (input.force_measurement_size() > 0) {
    CHECK_SIZE("force_measurement", nv, input.force_measurement_size());
    batch_estimator_.force_measurement.Set(input.force_measurement().data(), index);
  }

  // get force measurement
  double* force_measurement = batch_estimator_.force_measurement.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_measurement(force_measurement[i]);
  }

  // set force prediction
  if (input.force_prediction_size() > 0) {
    CHECK_SIZE("force_prediction", nv, input.force_prediction_size());
    batch_estimator_.force_prediction.Set(input.force_prediction().data(), index);
  }

  // get force prediction
  double* force_prediction = batch_estimator_.force_prediction.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_prediction(force_prediction[i]);
  }

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Settings(
    grpc::ServerContext* context, const batch_estimator::SettingsRequest* request,
    batch_estimator::SettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings
  batch_estimator::Settings input = request->settings();
  batch_estimator::Settings* output = response->mutable_settings();

  // configuration length
  if (input.has_configuration_length()) {
    // unpack
    int configuration_length = (int)(input.configuration_length());

    // check for valid length
    if (configuration_length < mjpc::MIN_HISTORY ||
        configuration_length > batch_estimator_.max_history) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid configuration length."};
    }

    // set
    batch_estimator_.SetConfigurationLength(configuration_length);
  }
  output->set_configuration_length(batch_estimator_.ConfigurationLength());

  // prior flag
  if (input.has_prior_flag())
    batch_estimator_.settings.prior_flag = input.prior_flag();
  output->set_prior_flag(batch_estimator_.settings.prior_flag);

  // sensor flag
  if (input.has_sensor_flag())
    batch_estimator_.settings.sensor_flag = input.sensor_flag();
  output->set_sensor_flag(batch_estimator_.settings.sensor_flag);

  // force flag
  if (input.has_force_flag())
    batch_estimator_.settings.force_flag = input.force_flag();
  output->set_force_flag(batch_estimator_.settings.force_flag);

  // max search iterations
  if (input.has_max_search_iterations()) {
    // unpack
    int iterations = input.max_search_iterations();

    // test valid
    if (iterations < 1) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid search iterations."};
    }

    // set
    batch_estimator_.settings.max_search_iterations = input.max_search_iterations();
  }
  output->set_max_search_iterations(batch_estimator_.settings.max_search_iterations);

  // max smoother iterations
  if (input.has_max_smoother_iterations()) {
    // unpack
    int iterations = input.max_smoother_iterations();

    // test valid
    if (iterations < 1) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid smoother iterations."};
    }

    // set
    batch_estimator_.settings.max_smoother_iterations =
        input.max_smoother_iterations();
  }
  output->set_max_smoother_iterations(
      batch_estimator_.settings.max_smoother_iterations);

  // gradient tolerance
  if (input.has_gradient_tolerance()) {
    batch_estimator_.settings.gradient_tolerance = input.gradient_tolerance();
  }
  output->set_gradient_tolerance(batch_estimator_.settings.gradient_tolerance);

  // verbose iteration
  if (input.has_verbose_iteration()) {
    batch_estimator_.settings.verbose_iteration = input.verbose_iteration();
  }
  output->set_verbose_iteration(batch_estimator_.settings.verbose_iteration);

  // verbose optimize
  if (input.has_verbose_optimize()) {
    batch_estimator_.settings.verbose_optimize = input.verbose_optimize();
  }
  output->set_verbose_optimize(batch_estimator_.settings.verbose_optimize);

  // verbose cost
  if (input.has_verbose_cost()) {
    batch_estimator_.settings.verbose_cost = input.verbose_cost();
  }
  output->set_verbose_cost(batch_estimator_.settings.verbose_cost);

  // verbose prior
  if (input.has_verbose_prior()) {
    batch_estimator_.settings.verbose_prior = input.verbose_prior();
  }
  output->set_verbose_prior(batch_estimator_.settings.verbose_prior);

  // band prior
  if (input.has_band_prior()) {
    batch_estimator_.settings.band_prior = input.band_prior();
  }
  output->set_band_prior(batch_estimator_.settings.band_prior);

  // search type
  if (input.has_search_type()) {
    // unpack
    mjpc::SearchType search_type = (mjpc::SearchType)(input.search_type());

    // check for valid search type
    if (search_type >= mjpc::kNumSearch) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
    }

    // set
    batch_estimator_.settings.search_type = search_type;
  }
  output->set_search_type((int)batch_estimator_.settings.search_type);

  // step scaling
  if (input.has_step_scaling()) {
    batch_estimator_.settings.step_scaling = input.step_scaling();
  }
  output->set_step_scaling(batch_estimator_.settings.step_scaling);

  // regularization initialization
  if (input.has_regularization_initial()) {
    batch_estimator_.settings.regularization_initial = input.regularization_initial();
  }
  output->set_regularization_initial(
      batch_estimator_.settings.regularization_initial);

  // regularization scaling
  if (input.has_regularization_scaling()) {
    batch_estimator_.settings.regularization_scaling = input.regularization_scaling();
  }
  output->set_regularization_scaling(
      batch_estimator_.settings.regularization_scaling);

  // band copy
  if (input.has_band_copy()) {
    batch_estimator_.settings.band_copy = input.band_copy();
  }
  output->set_band_copy(batch_estimator_.settings.band_copy);

  // time scaling
  if (input.has_time_scaling()) {
    batch_estimator_.settings.time_scaling = input.time_scaling();
  }
  output->set_time_scaling(batch_estimator_.settings.time_scaling);

  // search direction tolerance 
  if (input.has_search_direction_tolerance()) {
    batch_estimator_.settings.search_direction_tolerance = input.search_direction_tolerance();
  }
  output->set_search_direction_tolerance(batch_estimator_.settings.search_direction_tolerance);

  // cost tolerance 
  if (input.has_cost_tolerance()) {
    batch_estimator_.settings.cost_tolerance = input.cost_tolerance();
  }
  output->set_cost_tolerance(batch_estimator_.settings.cost_tolerance);

  // assemble prior Jacobian 
  if (input.has_assemble_prior_jacobian()) {
    batch_estimator_.settings.assemble_prior_jacobian = input.assemble_prior_jacobian();
  }
  output->set_assemble_prior_jacobian(batch_estimator_.settings.assemble_prior_jacobian);

  // assemble sensor Jacobian 
  if (input.has_assemble_sensor_jacobian()) {
    batch_estimator_.settings.assemble_sensor_jacobian = input.assemble_sensor_jacobian();
  }
  output->set_assemble_sensor_jacobian(batch_estimator_.settings.assemble_sensor_jacobian);

  // assemble force Jacobian 
  if (input.has_assemble_force_jacobian()) {
    batch_estimator_.settings.assemble_force_jacobian = input.assemble_force_jacobian();
  }
  output->set_assemble_force_jacobian(batch_estimator_.settings.assemble_force_jacobian);

  // assemble sensor norm hessian 
  if (input.has_assemble_sensor_norm_hessian()) {
    batch_estimator_.settings.assemble_sensor_norm_hessian = input.assemble_sensor_norm_hessian();
  }
  output->set_assemble_sensor_norm_hessian(batch_estimator_.settings.assemble_sensor_norm_hessian);

  // assemble force norm hessian 
  if (input.has_assemble_force_norm_hessian()) {
    batch_estimator_.settings.assemble_force_norm_hessian = input.assemble_force_norm_hessian();
  }
  output->set_assemble_force_norm_hessian(batch_estimator_.settings.assemble_force_norm_hessian);

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Cost(grpc::ServerContext* context,
                                    const batch_estimator::CostRequest* request,
                                    batch_estimator::CostResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  
  // cache settings 
  bool assemble_prior_jacobian = batch_estimator_.settings.assemble_prior_jacobian;
  bool assemble_sensor_jacobian = batch_estimator_.settings.assemble_sensor_jacobian;
  bool assemble_force_jacobian = batch_estimator_.settings.assemble_force_jacobian;
  bool assemble_sensor_norm_hessian = batch_estimator_.settings.assemble_sensor_norm_hessian;
  bool assemble_force_norm_hessian = batch_estimator_.settings.assemble_force_norm_hessian;

  if (request->internals()) {
    // compute dense cost internals
    batch_estimator_.settings.assemble_prior_jacobian = true;
    batch_estimator_.settings.assemble_sensor_jacobian = true;
    batch_estimator_.settings.assemble_force_jacobian = true;
    batch_estimator_.settings.assemble_sensor_norm_hessian = true;
    batch_estimator_.settings.assemble_force_norm_hessian = true;
  }

  // compute derivatives
  bool derivatives = request->derivatives();

  // evaluate cost
  batch_estimator_.cost = batch_estimator_.Cost(
      derivatives ? batch_estimator_.cost_gradient.data() : NULL,
      derivatives ? batch_estimator_.cost_hessian.data() : NULL, thread_pool_);

  // costs
  batch_estimator::Cost* cost = response->mutable_cost();

  // cost
  cost->set_total(batch_estimator_.cost);

  // prior cost
  cost->set_prior(batch_estimator_.cost_prior);

  // sensor cost
  cost->set_sensor(batch_estimator_.cost_sensor);

  // force cost
  cost->set_force(batch_estimator_.cost_force);

  // initial cost
  cost->set_initial(batch_estimator_.cost_initial);

  // derivatives 
  if (derivatives) {
    // dimension 
    int nvar = batch_estimator_.model->nv * batch_estimator_.ConfigurationLength();

    // unpack 
    double* gradient = batch_estimator_.cost_gradient.data();
    double* hessian = batch_estimator_.cost_hessian.data();

    // set gradient, Hessian
    for (int i = 0; i < nvar; i++) {
      cost->add_gradient(gradient[i]);
      for (int j = 0; j < nvar; j++) {
        cost->add_hessian(hessian[i * nvar + j]);
      }
    }
  }

  // dimensions
  int nv = batch_estimator_.model->nv, ns = batch_estimator_.SensorDimension();
  int nvar = nv * batch_estimator_.ConfigurationLength();
  int nsensor = ns * batch_estimator_.ConfigurationLength() - 1;
  int nforce = nv * batch_estimator_.ConfigurationLength() - 2;

  // set dimensions
  cost->set_nvar(nvar);
  cost->set_nsensor(nsensor);
  cost->set_nforce(nforce);

  // internals
  if (request->internals()) {
    // residual prior 
    const double* residual_prior = batch_estimator_.GetResidualPrior();
    for (int i = 0; i < nvar; i++) {
      cost->add_residual_prior(residual_prior[i]);
    }

    // residual sensor 
    const double* residual_sensor = batch_estimator_.GetResidualSensor();
    for (int i = 0; i < nsensor; i++) {
      cost->add_residual_sensor(residual_sensor[i]);
    }

    // residual force 
    const double* residual_force = batch_estimator_.GetResidualForce();
    for (int i = 0; i < nforce; i++) {
      cost->add_residual_force(residual_force[i]);
    }

    // Jacobian prior 
    const double* jacobian_prior = batch_estimator_.GetJacobianPrior();
    for (int i = 0; i < nvar; i++) {
      for (int j = 0; j < nvar; j++) {
        cost->add_jacobian_prior(jacobian_prior[i * nvar + j]);
      }
    }

    // Jacobian sensor 
    const double* jacobian_sensor = batch_estimator_.GetJacobianSensor();
    for (int i = 0; i < nsensor; i++) {
      for (int j = 0; j < nvar; j++) {
        cost->add_jacobian_sensor(jacobian_sensor[i * nvar + j]);
      }
    }

    // Jacobian force 
    const double* jacobian_force = batch_estimator_.GetJacobianForce();
    for (int i = 0; i < nforce; i++) {
      for (int j = 0; j < nvar; j++) {
        cost->add_jacobian_force(jacobian_force[i * nvar + j]);
      }
    }

    // norm gradient sensor 
    const double* norm_gradient_sensor = batch_estimator_.GetNormGradientSensor();
    for (int i = 0; i < nsensor; i++) {
      cost->add_norm_gradient_sensor(norm_gradient_sensor[i]);
    }

    // norm gradient force 
    const double* norm_gradient_force = batch_estimator_.GetNormGradientForce();
    for (int i = 0; i < nforce; i++) {
      cost->add_norm_gradient_force(norm_gradient_force[i]);
    }

    // prior matrix 
    const double* prior_matrix = batch_estimator_.weight_prior.data();
    for (int i = 0; i < nvar; i++) {
      for (int j = 0; j < nvar; j++) {
        cost->add_prior_matrix(prior_matrix[i * nvar + j]);
      }
    }

    // norm Hessian sensor 
    const double* norm_hessian_sensor = batch_estimator_.GetNormHessianSensor();
    for (int i = 0; i < nsensor; i++) {
      for (int j = 0; j < nsensor; j++) {
        cost->add_norm_hessian_sensor(norm_hessian_sensor[i * nsensor + j]);
      }
    }

    // norm Hessian force 
    const double* norm_hessian_force = batch_estimator_.GetNormHessianForce();
    for (int i = 0; i < nforce; i++) {
      for (int j = 0; j < nforce; j++) {
        cost->add_norm_hessian_force(norm_hessian_force[i * nforce + j]);
      }
    }
    
    // reset settings
    batch_estimator_.settings.assemble_prior_jacobian = assemble_prior_jacobian;
    batch_estimator_.settings.assemble_sensor_jacobian = assemble_sensor_jacobian;
    batch_estimator_.settings.assemble_force_jacobian = assemble_force_jacobian;
    batch_estimator_.settings.assemble_sensor_norm_hessian = assemble_sensor_norm_hessian;
    batch_estimator_.settings.assemble_force_norm_hessian = assemble_force_norm_hessian;
  }

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Weights(grpc::ServerContext* context,
                                       const batch_estimator::WeightsRequest* request,
                                       batch_estimator::WeightsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // weight
  batch_estimator::Weight input = request->weight();
  batch_estimator::Weight* output = response->mutable_weight();

  // prior
  if (input.has_prior()) {
    batch_estimator_.scale_prior = input.prior();
  }
  output->set_prior(batch_estimator_.scale_prior);

  // sensor
  int num_sensor = batch_estimator_.NumberSensors();
  if (input.sensor_size() > 0) {
    CHECK_SIZE("noise sensor", num_sensor, input.sensor_size());
    batch_estimator_.noise_sensor.assign(input.sensor().begin(),
                                   input.sensor().end());
  }
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor(batch_estimator_.noise_sensor[i]);
  }

  // force
  int nv = batch_estimator_.model->nv;
  if (input.force_size() > 0) {
    CHECK_SIZE("noise process", nv, input.force_size());
    batch_estimator_.noise_process.assign(input.force().begin(), input.force().end());
  }
  for (int i = 0; i < nv; i++) {
    output->add_force(batch_estimator_.noise_process[i]);
  }

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Norms(grpc::ServerContext* context,
                                     const batch_estimator::NormRequest* request,
                                     batch_estimator::NormResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // norm
  batch_estimator::Norm input = request->norm();
  batch_estimator::Norm* output = response->mutable_norm();

  // set sensor type
  int num_sensor = batch_estimator_.NumberSensors();
  if (input.sensor_type_size() > 0) {
    CHECK_SIZE("sensor_type", num_sensor, input.sensor_type_size());
    batch_estimator_.norm_type_sensor.clear();
    batch_estimator_.norm_type_sensor.reserve(num_sensor);
    for (const auto& sensor_type : input.sensor_type()) {
      batch_estimator_.norm_type_sensor.push_back(
          static_cast<mjpc::NormType>(sensor_type));
    }
  }

  // get sensor type
  for (const auto& sensor_type : batch_estimator_.norm_type_sensor) {
    output->add_sensor_type(sensor_type);
  }

  // set sensor parameters
  if (input.sensor_parameters_size() > 0) {
    CHECK_SIZE("sensor_parameters", mjpc::MAX_NORM_PARAMETERS * num_sensor,
               input.sensor_parameters_size());
    batch_estimator_.norm_parameters_sensor.assign(input.sensor_parameters().begin(),
                                             input.sensor_parameters().end());
  }

  // get sensor parameters
  for (const auto& sensor_parameters : batch_estimator_.norm_parameters_sensor) {
    output->add_sensor_parameters(sensor_parameters);
  }

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Shift(grpc::ServerContext* context,
                                     const batch_estimator::ShiftRequest* request,
                                     batch_estimator::ShiftResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // shift
  batch_estimator_.Shift(request->shift());

  // get head index
  response->set_head(batch_estimator_.configuration.Head());

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Reset(grpc::ServerContext* context,
                                     const batch_estimator::ResetRequest* request,
                                     batch_estimator::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  batch_estimator_.Reset();

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Optimize(
    grpc::ServerContext* context, const batch_estimator::OptimizeRequest* request,
    batch_estimator::OptimizeResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // optimize
  batch_estimator_.Optimize(thread_pool_);

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Status(grpc::ServerContext* context,
                                      const batch_estimator::StatusRequest* request,
                                      batch_estimator::StatusResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // status
  batch_estimator::Status* status = response->mutable_status();

  // search iterations
  status->set_search_iterations(batch_estimator_.IterationsSearch());

  // smoother iterations
  status->set_smoother_iterations(batch_estimator_.IterationsSmoother());

  // step size
  status->set_step_size(batch_estimator_.StepSize());

  // regularization
  status->set_regularization(batch_estimator_.Regularization());

  // gradient norm
  status->set_gradient_norm(batch_estimator_.GradientNorm());

  // search direction norm 
  status->set_search_direction_norm(batch_estimator_.SearchDirectionNorm());

  // solve status 
  status->set_solve_status((int)batch_estimator_.SolveStatus());

  // cost difference 
  status->set_cost_difference(batch_estimator_.CostDifference());

  // improvement 
  status->set_improvement(batch_estimator_.Improvement());

  // expected
  status->set_expected(batch_estimator_.Expected());

  // reduction ratio 
  status->set_reduction_ratio(batch_estimator_.ReductionRatio());

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::Timing(grpc::ServerContext* context,
                                      const batch_estimator::TimingRequest* request,
                                      batch_estimator::TimingResponse* response) {
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
  // double timer_update = 31;

  return grpc::Status::OK;
}

grpc::Status BatchEstimatorService::PriorMatrix(
    grpc::ServerContext* context, const batch_estimator::PriorMatrixRequest* request,
    batch_estimator::PriorMatrixResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // dimension
  int dim = batch_estimator_.model->nv * batch_estimator_.ConfigurationLength();
  response->set_dimension(dim);

  // set prior matrix
  // TODO(taylor): loop over upper triangle only
  if (request->prior_size() > 0) {
    CHECK_SIZE("prior_matrix", dim * dim, request->prior_size());
    batch_estimator_.weight_prior.assign(request->prior().begin(),
                                   request->prior().end());
  }

  // get prior matrix
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      response->add_prior(batch_estimator_.weight_prior[dim * i + j]);
    }
  }

  return grpc::Status::OK;
}

#undef CHECK_SIZE

}  // namespace batch_estimator_grpc
