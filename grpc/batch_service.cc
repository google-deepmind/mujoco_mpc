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

#include "grpc/batch_service.h"

#include <cstring>
#include <memory>
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

#include "grpc/batch.pb.h"
#include "mjpc/estimators/batch.h"
#include "mjpc/norm.h"

namespace mjpc::batch_grpc {

// TODO(taylor): make CheckSize utility function for agent and batch
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

BatchService::~BatchService() {}

grpc::Status BatchService::Init(grpc::ServerContext* context,
                                const batch::InitRequest* request,
                                batch::InitResponse* response) {
  // check configuration length
  if (request->configuration_length() < mjpc::kMinBatchHistory) {
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
  batch_model_override_ = std::move(tmp_model);

  // initialize batch
  int length = request->configuration_length();
  batch_.SetMaxHistory(length);
  batch_.Initialize(batch_model_override_.get());
  batch_.SetConfigurationLength(length);
  batch_.Reset();

  return grpc::Status::OK;
}

grpc::Status BatchService::Data(grpc::ServerContext* context,
                                const batch::DataRequest* request,
                                batch::DataResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // valid index
  int index = static_cast<int>(request->index());
  if (index < 0 || index >= batch_.ConfigurationLength()) {
    return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
  }

  // data
  batch::Data input = request->data();
  batch::Data* output = response->mutable_data();

  // set configuration
  int nq = batch_.model->nq;
  if (input.configuration_size() > 0) {
    CHECK_SIZE("configuration", nq, input.configuration_size());
    batch_.configuration.Set(input.configuration().data(), index);
  }

  // get configuration
  double* configuration = batch_.configuration.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration(configuration[i]);
  }

  // set velocity
  int nv = batch_.model->nv;
  if (input.velocity_size() > 0) {
    CHECK_SIZE("velocity", nv, input.velocity_size());
    batch_.velocity.Set(input.velocity().data(), index);
  }

  // get velocity
  double* velocity = batch_.velocity.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_velocity(velocity[i]);
  }

  // set acceleration
  if (input.acceleration_size() > 0) {
    CHECK_SIZE("acceleration", nv, input.acceleration_size());
    batch_.acceleration.Set(input.acceleration().data(), index);
  }

  // get acceleration
  double* acceleration = batch_.acceleration.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_acceleration(acceleration[i]);
  }

  // set time
  if (input.time_size() > 0) {
    CHECK_SIZE("time", 1, input.time_size());
    batch_.times.Set(input.time().data(), index);
  }

  // get time
  double* time = batch_.times.Get(index);
  output->add_time(time[0]);

  // set ctrl
  int nu = batch_.model->nu;
  if (input.ctrl_size() > 0) {
    CHECK_SIZE("ctrl", nu, input.ctrl_size());
    batch_.ctrl.Set(input.ctrl().data(), index);
  }

  // get ctrl
  double* ctrl = batch_.ctrl.Get(index);
  for (int i = 0; i < nu; i++) {
    output->add_ctrl(ctrl[i]);
  }

  // set previous configuration
  if (input.configuration_previous_size() > 0) {
    CHECK_SIZE("configuration_previous", nq,
               input.configuration_previous_size());
    batch_.configuration_previous.Set(input.configuration_previous().data(),
                                      index);
  }

  // get configuration previous
  double* prior = batch_.configuration_previous.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration_previous(prior[i]);
  }

  // set sensor measurement
  int ns = batch_.DimensionSensor();
  if (input.sensor_measurement_size() > 0) {
    CHECK_SIZE("sensor_measurement", ns, input.sensor_measurement_size());
    batch_.sensor_measurement.Set(input.sensor_measurement().data(), index);
  }

  // get sensor measurement
  double* sensor_measurement = batch_.sensor_measurement.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_measurement(sensor_measurement[i]);
  }

  // set sensor prediction
  if (input.sensor_prediction_size() > 0) {
    CHECK_SIZE("sensor_prediction", ns, input.sensor_prediction_size());
    batch_.sensor_prediction.Set(input.sensor_prediction().data(), index);
  }

  // get sensor prediction
  double* sensor_prediction = batch_.sensor_prediction.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_prediction(sensor_prediction[i]);
  }

  // set sensor mask
  int num_sensor = batch_.NumberSensors();
  if (input.sensor_mask_size() > 0) {
    CHECK_SIZE("sensor_mask", num_sensor, input.sensor_mask_size());
    batch_.sensor_mask.Set(input.sensor_mask().data(), index);
  }

  // get sensor mask
  int* sensor_mask = batch_.sensor_mask.Get(index);
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor_mask(sensor_mask[i]);
  }

  // set force measurement
  if (input.force_measurement_size() > 0) {
    CHECK_SIZE("force_measurement", nv, input.force_measurement_size());
    batch_.force_measurement.Set(input.force_measurement().data(), index);
  }

  // get force measurement
  double* force_measurement = batch_.force_measurement.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_measurement(force_measurement[i]);
  }

  // set force prediction
  if (input.force_prediction_size() > 0) {
    CHECK_SIZE("force_prediction", nv, input.force_prediction_size());
    batch_.force_prediction.Set(input.force_prediction().data(), index);
  }

  // get force prediction
  double* force_prediction = batch_.force_prediction.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_prediction(force_prediction[i]);
  }

  // parameters
  int np = batch_.NumberParameters();
  if (np > 0) {
    // set parameters
    if (input.parameters_size() > 0) {
      CHECK_SIZE("parameters", np, input.parameters_size());
      mju_copy(batch_.parameters.data(), input.parameters().data(), np);
    }

    // get parameters
    double* parameters = batch_.parameters.data();
    for (int i = 0; i < np; i++) {
      output->add_parameters(parameters[i]);
    }

    // set parameters previous
    if (input.parameters_previous_size() > 0) {
      CHECK_SIZE("parameters previous", np, input.parameters_previous_size());
      mju_copy(batch_.parameters_previous.data(),
               input.parameters_previous().data(), np);
    }

    // get parameters previous
    double* parameters_previous = batch_.parameters_previous.data();
    for (int i = 0; i < np; i++) {
      output->add_parameters_previous(parameters_previous[i]);
    }
  }

  return grpc::Status::OK;
}

grpc::Status BatchService::Settings(grpc::ServerContext* context,
                                    const batch::SettingsRequest* request,
                                    batch::SettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings
  batch::Settings input = request->settings();
  batch::Settings* output = response->mutable_settings();

  // configuration length
  if (input.has_configuration_length()) {
    // unpack
    int configuration_length = static_cast<int>(input.configuration_length());

    // check for valid length
    if (configuration_length < mjpc::kMinBatchHistory ||
        configuration_length > batch_.GetMaxHistory()) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid configuration length."};
    }

    // set
    batch_.SetConfigurationLength(configuration_length);
  }
  output->set_configuration_length(batch_.ConfigurationLength());

  // prior flag
  if (input.has_prior_flag()) batch_.settings.prior_flag = input.prior_flag();
  output->set_prior_flag(batch_.settings.prior_flag);

  // sensor flag
  if (input.has_sensor_flag())
    batch_.settings.sensor_flag = input.sensor_flag();
  output->set_sensor_flag(batch_.settings.sensor_flag);

  // force flag
  if (input.has_force_flag()) batch_.settings.force_flag = input.force_flag();
  output->set_force_flag(batch_.settings.force_flag);

  // max search iterations
  if (input.has_max_search_iterations()) {
    // unpack
    int iterations = input.max_search_iterations();

    // test valid
    if (iterations < 1) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid search iterations."};
    }

    // set
    batch_.settings.max_search_iterations = input.max_search_iterations();
  }
  output->set_max_search_iterations(batch_.settings.max_search_iterations);

  // max smoother iterations
  if (input.has_max_smoother_iterations()) {
    // unpack
    int iterations = input.max_smoother_iterations();

    // test valid
    if (iterations < 1) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid smoother iterations."};
    }

    // set
    batch_.settings.max_smoother_iterations = input.max_smoother_iterations();
  }
  output->set_max_smoother_iterations(batch_.settings.max_smoother_iterations);

  // gradient tolerance
  if (input.has_gradient_tolerance()) {
    batch_.settings.gradient_tolerance = input.gradient_tolerance();
  }
  output->set_gradient_tolerance(batch_.settings.gradient_tolerance);

  // verbose iteration
  if (input.has_verbose_iteration()) {
    batch_.settings.verbose_iteration = input.verbose_iteration();
  }
  output->set_verbose_iteration(batch_.settings.verbose_iteration);

  // verbose optimize
  if (input.has_verbose_optimize()) {
    batch_.settings.verbose_optimize = input.verbose_optimize();
  }
  output->set_verbose_optimize(batch_.settings.verbose_optimize);

  // verbose cost
  if (input.has_verbose_cost()) {
    batch_.settings.verbose_cost = input.verbose_cost();
  }
  output->set_verbose_cost(batch_.settings.verbose_cost);

  // verbose prior
  if (input.has_verbose_prior()) {
    batch_.settings.verbose_prior = input.verbose_prior();
  }
  output->set_verbose_prior(batch_.settings.verbose_prior);

  // search type
  if (input.has_search_type()) {
    // unpack
    mjpc::SearchType search_type = (mjpc::SearchType)(input.search_type());

    // check for valid search type
    if (search_type >= mjpc::kNumSearch) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
    }

    // set
    batch_.settings.search_type = search_type;
  }
  output->set_search_type(static_cast<int>(batch_.settings.search_type));

  // step scaling
  if (input.has_step_scaling()) {
    batch_.settings.step_scaling = input.step_scaling();
  }
  output->set_step_scaling(batch_.settings.step_scaling);

  // regularization initialization
  if (input.has_regularization_initial()) {
    batch_.settings.regularization_initial = input.regularization_initial();
  }
  output->set_regularization_initial(batch_.settings.regularization_initial);

  // regularization scaling
  if (input.has_regularization_scaling()) {
    batch_.settings.regularization_scaling = input.regularization_scaling();
  }
  output->set_regularization_scaling(batch_.settings.regularization_scaling);

  // time scaling (force)
  if (input.has_time_scaling_force()) {
    batch_.settings.time_scaling_force = input.time_scaling_force();
  }
  output->set_time_scaling_force(batch_.settings.time_scaling_force);

  // time scaling (sensor)
  if (input.has_time_scaling_sensor()) {
    batch_.settings.time_scaling_sensor = input.time_scaling_sensor();
  }
  output->set_time_scaling_sensor(batch_.settings.time_scaling_sensor);

  // search direction tolerance
  if (input.has_search_direction_tolerance()) {
    batch_.settings.search_direction_tolerance =
        input.search_direction_tolerance();
  }
  output->set_search_direction_tolerance(
      batch_.settings.search_direction_tolerance);

  // cost tolerance
  if (input.has_cost_tolerance()) {
    batch_.settings.cost_tolerance = input.cost_tolerance();
  }
  output->set_cost_tolerance(batch_.settings.cost_tolerance);

  // assemble prior Jacobian
  if (input.has_assemble_prior_jacobian()) {
    batch_.settings.assemble_prior_jacobian = input.assemble_prior_jacobian();
  }
  output->set_assemble_prior_jacobian(batch_.settings.assemble_prior_jacobian);

  // assemble sensor Jacobian
  if (input.has_assemble_sensor_jacobian()) {
    batch_.settings.assemble_sensor_jacobian = input.assemble_sensor_jacobian();
  }
  output->set_assemble_sensor_jacobian(
      batch_.settings.assemble_sensor_jacobian);

  // assemble force Jacobian
  if (input.has_assemble_force_jacobian()) {
    batch_.settings.assemble_force_jacobian = input.assemble_force_jacobian();
  }
  output->set_assemble_force_jacobian(batch_.settings.assemble_force_jacobian);

  // assemble sensor norm hessian
  if (input.has_assemble_sensor_norm_hessian()) {
    batch_.settings.assemble_sensor_norm_hessian =
        input.assemble_sensor_norm_hessian();
  }
  output->set_assemble_sensor_norm_hessian(
      batch_.settings.assemble_sensor_norm_hessian);

  // assemble force norm hessian
  if (input.has_assemble_force_norm_hessian()) {
    batch_.settings.assemble_force_norm_hessian =
        input.assemble_force_norm_hessian();
  }
  output->set_assemble_force_norm_hessian(
      batch_.settings.assemble_force_norm_hessian);

  // first step position sensors
  if (input.has_first_step_position_sensors()) {
    batch_.settings.first_step_position_sensors =
        input.first_step_position_sensors();
  }
  output->set_first_step_position_sensors(
      batch_.settings.first_step_position_sensors);

  // last step position sensors
  if (input.has_last_step_position_sensors()) {
    batch_.settings.last_step_position_sensors =
        input.last_step_position_sensors();
  }
  output->set_last_step_position_sensors(
      batch_.settings.last_step_position_sensors);

  // last step velocity sensors
  if (input.has_last_step_velocity_sensors()) {
    batch_.settings.last_step_velocity_sensors =
        input.last_step_velocity_sensors();
  }
  output->set_last_step_velocity_sensors(
      batch_.settings.last_step_velocity_sensors);

  return grpc::Status::OK;
}

grpc::Status BatchService::Cost(grpc::ServerContext* context,
                                const batch::CostRequest* request,
                                batch::CostResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // compute derivatives
  bool derivatives = request->derivatives();

  // evaluate cost
  double total_cost =
      batch_.Cost(derivatives ? batch_.GetCostGradient() : NULL,
                  derivatives ? batch_.GetCostHessianBand() : NULL);

  // cost
  response->set_total(total_cost);

  // prior cost
  response->set_prior(batch_.GetCostPrior());

  // sensor cost
  response->set_sensor(batch_.GetCostSensor());

  // force cost
  response->set_force(batch_.GetCostForce());

  // initial cost
  response->set_initial(batch_.GetCostInitial());

  // derivatives
  if (derivatives) {
    // dimension
    int nvar = batch_.model->nv * batch_.ConfigurationLength();

    // unpack
    double* gradient = batch_.GetCostGradient();
    double* hessian = batch_.GetCostHessian();

    // set gradient, Hessian
    for (int i = 0; i < nvar; i++) {
      response->add_gradient(gradient[i]);
      for (int j = 0; j < nvar; j++) {
        response->add_hessian(hessian[i * nvar + j]);
      }
    }
  }

  // dimensions
  int nv = batch_.model->nv, ns = batch_.DimensionSensor();
  int nvar = nv * batch_.ConfigurationLength();
  int nsensor_ = ns * (batch_.ConfigurationLength() - 1);
  int nforce = nv * (batch_.ConfigurationLength() - 2);

  // set dimensions
  response->set_nvar(nvar);
  response->set_nsensor(nsensor_);
  response->set_nforce(nforce);

  // internals
  if (request->internals()) {
    // residual prior
    const double* residual_prior = batch_.GetResidualPrior();
    for (int i = 0; i < nvar; i++) {
      response->add_residual_prior(residual_prior[i]);
    }

    // residual sensor
    const double* residual_sensor = batch_.GetResidualSensor();
    for (int i = 0; i < nsensor_; i++) {
      response->add_residual_sensor(residual_sensor[i]);
    }

    // residual force
    const double* residual_force = batch_.GetResidualForce();
    for (int i = 0; i < nforce; i++) {
      response->add_residual_force(residual_force[i]);
    }

    // Jacobian prior
    const double* jacobian_prior = batch_.GetJacobianPrior();
    for (int i = 0; i < nvar; i++) {
      for (int j = 0; j < nvar; j++) {
        response->add_jacobian_prior(jacobian_prior[i * nvar + j]);
      }
    }

    // Jacobian sensor
    const double* jacobian_sensor = batch_.GetJacobianSensor();
    for (int i = 0; i < nsensor_; i++) {
      for (int j = 0; j < nvar; j++) {
        response->add_jacobian_sensor(jacobian_sensor[i * nvar + j]);
      }
    }

    // Jacobian force
    const double* jacobian_force = batch_.GetJacobianForce();
    for (int i = 0; i < nforce; i++) {
      for (int j = 0; j < nvar; j++) {
        response->add_jacobian_force(jacobian_force[i * nvar + j]);
      }
    }

    // norm gradient sensor
    const double* norm_gradient_sensor = batch_.GetNormGradientSensor();
    for (int i = 0; i < nsensor_; i++) {
      response->add_norm_gradient_sensor(norm_gradient_sensor[i]);
    }

    // norm gradient force
    const double* norm_gradient_force = batch_.GetNormGradientForce();
    for (int i = 0; i < nforce; i++) {
      response->add_norm_gradient_force(norm_gradient_force[i]);
    }

    // prior matrix
    const double* prior_matrix = batch_.PriorWeights();
    for (int i = 0; i < nvar; i++) {
      for (int j = 0; j < nvar; j++) {
        response->add_prior_matrix(
            batch_.settings.prior_flag ? prior_matrix[i * nvar + j] : 0.0);
      }
    }

    // norm Hessian sensor
    const double* norm_hessian_sensor = batch_.GetNormHessianSensor();
    for (int i = 0; i < nsensor_; i++) {
      for (int j = 0; j < nsensor_; j++) {
        response->add_norm_hessian_sensor(
            norm_hessian_sensor[i * nsensor_ + j]);
      }
    }

    // norm Hessian force
    const double* norm_hessian_force = batch_.GetNormHessianForce();
    for (int i = 0; i < nforce; i++) {
      for (int j = 0; j < nforce; j++) {
        response->add_norm_hessian_force(norm_hessian_force[i * nforce + j]);
      }
    }
  }

  return grpc::Status::OK;
}

grpc::Status BatchService::Noise(grpc::ServerContext* context,
                                 const batch::NoiseRequest* request,
                                 batch::NoiseResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // weight
  batch::Noise input = request->noise();
  batch::Noise* output = response->mutable_noise();

  // process
  int nv = batch_.model->nv;
  if (input.process_size() > 0) {
    CHECK_SIZE("noise process", nv, input.process_size());
    batch_.noise_process.assign(input.process().begin(), input.process().end());
  }
  for (int i = 0; i < nv; i++) {
    output->add_process(batch_.noise_process[i]);
  }

  // sensor
  int num_sensor = batch_.NumberSensors();
  if (input.sensor_size() > 0) {
    CHECK_SIZE("noise sensor", num_sensor, input.sensor_size());
    batch_.noise_sensor.assign(input.sensor().begin(), input.sensor().end());
  }
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor(batch_.noise_sensor[i]);
  }

  // parameters
  int num_parameters = batch_.NumberParameters();
  if (num_parameters > 0) {
    if (input.parameter_size() > 0) {
      CHECK_SIZE("noise parameter", num_parameters, input.parameter_size());
      batch_.noise_parameter.assign(input.parameter().begin(),
                                    input.parameter().end());
    }
    for (int i = 0; i < num_parameters; i++) {
      output->add_parameter(batch_.noise_parameter[i]);
    }
  }

  return grpc::Status::OK;
}

grpc::Status BatchService::Norms(grpc::ServerContext* context,
                                 const batch::NormRequest* request,
                                 batch::NormResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // norm
  batch::Norm input = request->norm();
  batch::Norm* output = response->mutable_norm();

  // set sensor type
  int num_sensor = batch_.NumberSensors();
  if (input.sensor_type_size() > 0) {
    CHECK_SIZE("sensor_type", num_sensor, input.sensor_type_size());
    batch_.norm_type_sensor.clear();
    batch_.norm_type_sensor.reserve(num_sensor);
    for (const auto& sensor_type : input.sensor_type()) {
      batch_.norm_type_sensor.push_back(
          static_cast<mjpc::NormType>(sensor_type));
    }
  }

  // get sensor type
  for (const auto& sensor_type : batch_.norm_type_sensor) {
    output->add_sensor_type(sensor_type);
  }

  // set sensor parameters
  if (input.sensor_parameters_size() > 0) {
    CHECK_SIZE("sensor_parameters", mjpc::kMaxNormParameters * num_sensor,
               input.sensor_parameters_size());
    batch_.norm_parameters_sensor.assign(input.sensor_parameters().begin(),
                                         input.sensor_parameters().end());
  }

  // get sensor parameters
  for (const auto& sensor_parameters : batch_.norm_parameters_sensor) {
    output->add_sensor_parameters(sensor_parameters);
  }

  return grpc::Status::OK;
}

grpc::Status BatchService::Shift(grpc::ServerContext* context,
                                 const batch::ShiftRequest* request,
                                 batch::ShiftResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // shift
  batch_.Shift(request->shift());

  // get head index
  response->set_head(batch_.configuration.Head());

  return grpc::Status::OK;
}

grpc::Status BatchService::Reset(grpc::ServerContext* context,
                                 const batch::ResetRequest* request,
                                 batch::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  batch_.Reset();

  return grpc::Status::OK;
}

grpc::Status BatchService::Optimize(grpc::ServerContext* context,
                                    const batch::OptimizeRequest* request,
                                    batch::OptimizeResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // optimize
  batch_.Optimize();

  return grpc::Status::OK;
}

grpc::Status BatchService::Status(grpc::ServerContext* context,
                                  const batch::StatusRequest* request,
                                  batch::StatusResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // status
  batch::Status* status = response->mutable_status();

  // search iterations
  status->set_search_iterations(batch_.IterationsSearch());

  // smoother iterations
  status->set_smoother_iterations(batch_.IterationsSmoother());

  // step size
  status->set_step_size(batch_.StepSize());

  // regularization
  status->set_regularization(batch_.Regularization());

  // gradient norm
  status->set_gradient_norm(batch_.GradientNorm());

  // search direction norm
  status->set_search_direction_norm(batch_.SearchDirectionNorm());

  // solve status
  status->set_solve_status(static_cast<int>(batch_.SolveStatus()));

  // cost difference
  status->set_cost_difference(batch_.CostDifference());

  // improvement
  status->set_improvement(batch_.Improvement());

  // expected
  status->set_expected(batch_.Expected());

  // reduction ratio
  status->set_reduction_ratio(batch_.ReductionRatio());

  return grpc::Status::OK;
}

grpc::Status BatchService::Timing(grpc::ServerContext* context,
                                  const batch::TimingRequest* request,
                                  batch::TimingResponse* response) {
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

grpc::Status BatchService::PriorWeights(
    grpc::ServerContext* context, const batch::PriorWeightsRequest* request,
    batch::PriorWeightsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // dimension
  int dim = batch_.model->nv * batch_.ConfigurationLength();
  response->set_dimension(dim);

  // set prior matrix
  if (request->weights_size() > 0) {
    CHECK_SIZE("prior weights", dim * dim, request->weights_size());
    batch_.SetPriorWeights(request->weights().data());
  }

  // get prior matrix
  const double* weights = batch_.PriorWeights();
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      response->add_weights(batch_.settings.prior_flag ? weights[dim * i + j]
                                                       : 0.0);
    }
  }

  return grpc::Status::OK;
}

grpc::Status BatchService::SensorInfo(grpc::ServerContext* context,
                                      const batch::SensorInfoRequest* request,
                                      batch::SensorInfoResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // start index
  response->set_start_index(batch_.SensorStartIndex());

  // number of sensor measurements
  response->set_num_measurements(batch_.NumberSensors());

  // sensor measurement dimension
  response->set_dim_measurements(batch_.DimensionSensor());

  return grpc::Status::OK;
}

#undef CHECK_SIZE

}  // namespace mjpc::batch_grpc
