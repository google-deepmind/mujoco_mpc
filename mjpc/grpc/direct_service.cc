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

#include "mjpc/grpc/direct_service.h"

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

#include "mjpc/grpc/direct.pb.h"
#include "mjpc/direct/direct.h"

namespace mjpc::direct_grpc {

// TODO(taylor): make CheckSize utility function for agent and direct
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

DirectService::~DirectService() {}

grpc::Status DirectService::Init(grpc::ServerContext* context,
                                 const direct::InitRequest* request,
                                 direct::InitResponse* response) {
  // check configuration length
  if (request->configuration_length() < mjpc::kMinDirectHistory) {
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
  model_override_ = std::move(tmp_model);

  // initialize direct
  int length = request->configuration_length();
  optimizer_.SetMaxHistory(length);
  optimizer_.Initialize(model_override_.get());
  optimizer_.SetConfigurationLength(length);
  optimizer_.Reset();

  return grpc::Status::OK;
}

grpc::Status DirectService::Data(grpc::ServerContext* context,
                                 const direct::DataRequest* request,
                                 direct::DataResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // valid index
  int index = static_cast<int>(request->index());
  if (index < 0 || index >= optimizer_.ConfigurationLength()) {
    return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
  }

  // data
  direct::Data input = request->data();
  direct::Data* output = response->mutable_data();

  // set configuration
  int nq = optimizer_.model->nq;
  if (input.configuration_size() > 0) {
    CHECK_SIZE("configuration", nq, input.configuration_size());
    optimizer_.configuration.Set(input.configuration().data(), index);
  }

  // get configuration
  double* configuration = optimizer_.configuration.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration(configuration[i]);
  }

  // set velocity
  int nv = optimizer_.model->nv;
  if (input.velocity_size() > 0) {
    CHECK_SIZE("velocity", nv, input.velocity_size());
    optimizer_.velocity.Set(input.velocity().data(), index);
  }

  // get velocity
  double* velocity = optimizer_.velocity.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_velocity(velocity[i]);
  }

  // set acceleration
  if (input.acceleration_size() > 0) {
    CHECK_SIZE("acceleration", nv, input.acceleration_size());
    optimizer_.acceleration.Set(input.acceleration().data(), index);
  }

  // get acceleration
  double* acceleration = optimizer_.acceleration.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_acceleration(acceleration[i]);
  }

  // set time
  if (input.time_size() > 0) {
    CHECK_SIZE("time", 1, input.time_size());
    optimizer_.times.Set(input.time().data(), index);
  }

  // get time
  double* time = optimizer_.times.Get(index);
  output->add_time(time[0]);

  // set ctrl
  int nu = optimizer_.model->nu;
  if (input.ctrl_size() > 0) {
    CHECK_SIZE("ctrl", nu, input.ctrl_size());
    optimizer_.ctrl.Set(input.ctrl().data(), index);
  }

  // get ctrl
  double* ctrl = optimizer_.ctrl.Get(index);
  for (int i = 0; i < nu; i++) {
    output->add_ctrl(ctrl[i]);
  }

  // set previous configuration
  if (input.configuration_previous_size() > 0) {
    CHECK_SIZE("configuration_previous", nq,
               input.configuration_previous_size());
    optimizer_.configuration_previous.Set(input.configuration_previous().data(),
                                          index);
  }

  // get configuration previous
  double* qpos_prev = optimizer_.configuration_previous.Get(index);
  for (int i = 0; i < nq; i++) {
    output->add_configuration_previous(qpos_prev[i]);
  }

  // set sensor measurement
  int ns = optimizer_.DimensionSensor();
  if (input.sensor_measurement_size() > 0) {
    CHECK_SIZE("sensor_measurement", ns, input.sensor_measurement_size());
    optimizer_.sensor_measurement.Set(input.sensor_measurement().data(), index);
  }

  // get sensor measurement
  double* sensor_measurement = optimizer_.sensor_measurement.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_measurement(sensor_measurement[i]);
  }

  // set sensor prediction
  if (input.sensor_prediction_size() > 0) {
    CHECK_SIZE("sensor_prediction", ns, input.sensor_prediction_size());
    optimizer_.sensor_prediction.Set(input.sensor_prediction().data(), index);
  }

  // get sensor prediction
  double* sensor_prediction = optimizer_.sensor_prediction.Get(index);
  for (int i = 0; i < ns; i++) {
    output->add_sensor_prediction(sensor_prediction[i]);
  }

  // set sensor mask
  int num_sensor = optimizer_.NumberSensors();
  if (input.sensor_mask_size() > 0) {
    CHECK_SIZE("sensor_mask", num_sensor, input.sensor_mask_size());
    optimizer_.sensor_mask.Set(input.sensor_mask().data(), index);
  }

  // get sensor mask
  int* sensor_mask = optimizer_.sensor_mask.Get(index);
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor_mask(sensor_mask[i]);
  }

  // set force measurement
  if (input.force_measurement_size() > 0) {
    CHECK_SIZE("force_measurement", nv, input.force_measurement_size());
    optimizer_.force_measurement.Set(input.force_measurement().data(), index);
  }

  // get force measurement
  double* force_measurement = optimizer_.force_measurement.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_measurement(force_measurement[i]);
  }

  // set force prediction
  if (input.force_prediction_size() > 0) {
    CHECK_SIZE("force_prediction", nv, input.force_prediction_size());
    optimizer_.force_prediction.Set(input.force_prediction().data(), index);
  }

  // get force prediction
  double* force_prediction = optimizer_.force_prediction.Get(index);
  for (int i = 0; i < nv; i++) {
    output->add_force_prediction(force_prediction[i]);
  }

  // parameters
  int np = optimizer_.NumberParameters();
  if (np > 0) {
    // set parameters
    if (input.parameters_size() > 0) {
      CHECK_SIZE("parameters", np, input.parameters_size());
      mju_copy(optimizer_.parameters.data(), input.parameters().data(), np);
    }

    // get parameters
    double* parameters = optimizer_.parameters.data();
    for (int i = 0; i < np; i++) {
      output->add_parameters(parameters[i]);
    }

    // set parameters previous
    if (input.parameters_previous_size() > 0) {
      CHECK_SIZE("parameters previous", np, input.parameters_previous_size());
      mju_copy(optimizer_.parameters_previous.data(),
               input.parameters_previous().data(), np);
    }

    // get parameters previous
    double* parameters_previous = optimizer_.parameters_previous.data();
    for (int i = 0; i < np; i++) {
      output->add_parameters_previous(parameters_previous[i]);
    }
  }

  return grpc::Status::OK;
}

grpc::Status DirectService::Settings(grpc::ServerContext* context,
                                     const direct::SettingsRequest* request,
                                     direct::SettingsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // settings
  direct::Settings input = request->settings();
  direct::Settings* output = response->mutable_settings();

  // configuration length
  if (input.has_configuration_length()) {
    // unpack
    int configuration_length = static_cast<int>(input.configuration_length());

    // check for valid length
    if (configuration_length < mjpc::kMinDirectHistory ||
        configuration_length > optimizer_.GetMaxHistory()) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid configuration length."};
    }

    // set
    optimizer_.SetConfigurationLength(configuration_length);
  }
  output->set_configuration_length(optimizer_.ConfigurationLength());

  // sensor flag
  if (input.has_sensor_flag())
    optimizer_.settings.sensor_flag = input.sensor_flag();
  output->set_sensor_flag(optimizer_.settings.sensor_flag);

  // force flag
  if (input.has_force_flag())
    optimizer_.settings.force_flag = input.force_flag();
  output->set_force_flag(optimizer_.settings.force_flag);

  // max search iterations
  if (input.has_max_search_iterations()) {
    // unpack
    int iterations = input.max_search_iterations();

    // test valid
    if (iterations < 1) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid search iterations."};
    }

    // set
    optimizer_.settings.max_search_iterations = input.max_search_iterations();
  }
  output->set_max_search_iterations(optimizer_.settings.max_search_iterations);

  // max smoother iterations
  if (input.has_max_smoother_iterations()) {
    // unpack
    int iterations = input.max_smoother_iterations();

    // test valid
    if (iterations < 1) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid smoother iterations."};
    }

    // set
    optimizer_.settings.max_smoother_iterations =
        input.max_smoother_iterations();
  }
  output->set_max_smoother_iterations(
      optimizer_.settings.max_smoother_iterations);

  // gradient tolerance
  if (input.has_gradient_tolerance()) {
    optimizer_.settings.gradient_tolerance = input.gradient_tolerance();
  }
  output->set_gradient_tolerance(optimizer_.settings.gradient_tolerance);

  // verbose iteration
  if (input.has_verbose_iteration()) {
    optimizer_.settings.verbose_iteration = input.verbose_iteration();
  }
  output->set_verbose_iteration(optimizer_.settings.verbose_iteration);

  // verbose optimize
  if (input.has_verbose_optimize()) {
    optimizer_.settings.verbose_optimize = input.verbose_optimize();
  }
  output->set_verbose_optimize(optimizer_.settings.verbose_optimize);

  // verbose cost
  if (input.has_verbose_cost()) {
    optimizer_.settings.verbose_cost = input.verbose_cost();
  }
  output->set_verbose_cost(optimizer_.settings.verbose_cost);

  // search type
  if (input.has_search_type()) {
    // unpack
    mjpc::SearchType search_type = (mjpc::SearchType)(input.search_type());

    // check for valid search type
    if (search_type >= mjpc::kNumSearch) {
      return {grpc::StatusCode::OUT_OF_RANGE, "Invalid index."};
    }

    // set
    optimizer_.settings.search_type = search_type;
  }
  output->set_search_type(static_cast<int>(optimizer_.settings.search_type));

  // step scaling
  if (input.has_step_scaling()) {
    optimizer_.settings.step_scaling = input.step_scaling();
  }
  output->set_step_scaling(optimizer_.settings.step_scaling);

  // regularization initialization
  if (input.has_regularization_initial()) {
    optimizer_.settings.regularization_initial = input.regularization_initial();
  }
  output->set_regularization_initial(
      optimizer_.settings.regularization_initial);

  // regularization scaling
  if (input.has_regularization_scaling()) {
    optimizer_.settings.regularization_scaling = input.regularization_scaling();
  }
  output->set_regularization_scaling(
      optimizer_.settings.regularization_scaling);

  // time scaling (force)
  if (input.has_time_scaling_force()) {
    optimizer_.settings.time_scaling_force = input.time_scaling_force();
  }
  output->set_time_scaling_force(optimizer_.settings.time_scaling_force);

  // time scaling (sensor)
  if (input.has_time_scaling_sensor()) {
    optimizer_.settings.time_scaling_sensor = input.time_scaling_sensor();
  }
  output->set_time_scaling_sensor(optimizer_.settings.time_scaling_sensor);

  // search direction tolerance
  if (input.has_search_direction_tolerance()) {
    optimizer_.settings.search_direction_tolerance =
        input.search_direction_tolerance();
  }
  output->set_search_direction_tolerance(
      optimizer_.settings.search_direction_tolerance);

  // cost tolerance
  if (input.has_cost_tolerance()) {
    optimizer_.settings.cost_tolerance = input.cost_tolerance();
  }
  output->set_cost_tolerance(optimizer_.settings.cost_tolerance);

  // assemble sensor Jacobian
  if (input.has_assemble_sensor_jacobian()) {
    optimizer_.settings.assemble_sensor_jacobian =
        input.assemble_sensor_jacobian();
  }
  output->set_assemble_sensor_jacobian(
      optimizer_.settings.assemble_sensor_jacobian);

  // assemble force Jacobian
  if (input.has_assemble_force_jacobian()) {
    optimizer_.settings.assemble_force_jacobian =
        input.assemble_force_jacobian();
  }
  output->set_assemble_force_jacobian(
      optimizer_.settings.assemble_force_jacobian);

  // assemble sensor norm hessian
  if (input.has_assemble_sensor_norm_hessian()) {
    optimizer_.settings.assemble_sensor_norm_hessian =
        input.assemble_sensor_norm_hessian();
  }
  output->set_assemble_sensor_norm_hessian(
      optimizer_.settings.assemble_sensor_norm_hessian);

  // assemble force norm hessian
  if (input.has_assemble_force_norm_hessian()) {
    optimizer_.settings.assemble_force_norm_hessian =
        input.assemble_force_norm_hessian();
  }
  output->set_assemble_force_norm_hessian(
      optimizer_.settings.assemble_force_norm_hessian);

  // first step position sensors
  if (input.has_first_step_position_sensors()) {
    optimizer_.settings.first_step_position_sensors =
        input.first_step_position_sensors();
  }
  output->set_first_step_position_sensors(
      optimizer_.settings.first_step_position_sensors);

  // last step position sensors
  if (input.has_last_step_position_sensors()) {
    optimizer_.settings.last_step_position_sensors =
        input.last_step_position_sensors();
  }
  output->set_last_step_position_sensors(
      optimizer_.settings.last_step_position_sensors);

  // last step velocity sensors
  if (input.has_last_step_velocity_sensors()) {
    optimizer_.settings.last_step_velocity_sensors =
        input.last_step_velocity_sensors();
  }
  output->set_last_step_velocity_sensors(
      optimizer_.settings.last_step_velocity_sensors);

  return grpc::Status::OK;
}

grpc::Status DirectService::Cost(grpc::ServerContext* context,
                                 const direct::CostRequest* request,
                                 direct::CostResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // compute derivatives
  bool derivatives = request->derivatives();

  // evaluate cost
  double total_cost =
      optimizer_.Cost(derivatives ? optimizer_.GetCostGradient() : NULL,
                      derivatives ? optimizer_.GetCostHessianBand() : NULL);

  // cost
  response->set_total(total_cost);

  // sensor cost
  response->set_sensor(optimizer_.GetCostSensor());

  // force cost
  response->set_force(optimizer_.GetCostForce());

  // parameter cost
  response->set_parameter(optimizer_.GetCostParameter());

  // initial cost
  response->set_initial(optimizer_.GetCostInitial());

  // derivatives
  if (derivatives) {
    // dimension
    int nvar = optimizer_.model->nv * optimizer_.ConfigurationLength();

    // unpack
    double* gradient = optimizer_.GetCostGradient();
    double* hessian = optimizer_.GetCostHessian();

    // set gradient, Hessian
    for (int i = 0; i < nvar; i++) {
      response->add_gradient(gradient[i]);
      for (int j = 0; j < nvar; j++) {
        response->add_hessian(hessian[i * nvar + j]);
      }
    }
  }

  // dimensions
  int nv = optimizer_.model->nv, ns = optimizer_.DimensionSensor();
  int nvar = nv * optimizer_.ConfigurationLength();
  int nsensor_ = ns * (optimizer_.ConfigurationLength() - 1);
  int nforce = nv * (optimizer_.ConfigurationLength() - 2);

  // set dimensions
  response->set_nvar(nvar);
  response->set_nsensor(nsensor_);
  response->set_nforce(nforce);

  // internals
  if (request->internals()) {
    // residual sensor
    const double* residual_sensor = optimizer_.GetResidualSensor();
    for (int i = 0; i < nsensor_; i++) {
      response->add_residual_sensor(residual_sensor[i]);
    }

    // residual force
    const double* residual_force = optimizer_.GetResidualForce();
    for (int i = 0; i < nforce; i++) {
      response->add_residual_force(residual_force[i]);
    }

    // Jacobian sensor
    const double* jacobian_sensor = optimizer_.GetJacobianSensor();
    for (int i = 0; i < nsensor_; i++) {
      for (int j = 0; j < nvar; j++) {
        response->add_jacobian_sensor(jacobian_sensor[i * nvar + j]);
      }
    }

    // Jacobian force
    const double* jacobian_force = optimizer_.GetJacobianForce();
    for (int i = 0; i < nforce; i++) {
      for (int j = 0; j < nvar; j++) {
        response->add_jacobian_force(jacobian_force[i * nvar + j]);
      }
    }

    // norm gradient sensor
    const double* norm_gradient_sensor = optimizer_.GetNormGradientSensor();
    for (int i = 0; i < nsensor_; i++) {
      response->add_norm_gradient_sensor(norm_gradient_sensor[i]);
    }

    // norm gradient force
    const double* norm_gradient_force = optimizer_.GetNormGradientForce();
    for (int i = 0; i < nforce; i++) {
      response->add_norm_gradient_force(norm_gradient_force[i]);
    }

    // norm Hessian sensor
    const double* norm_hessian_sensor = optimizer_.GetNormHessianSensor();
    for (int i = 0; i < nsensor_; i++) {
      for (int j = 0; j < nsensor_; j++) {
        response->add_norm_hessian_sensor(
            norm_hessian_sensor[i * nsensor_ + j]);
      }
    }

    // norm Hessian force
    const double* norm_hessian_force = optimizer_.GetNormHessianForce();
    for (int i = 0; i < nforce; i++) {
      for (int j = 0; j < nforce; j++) {
        response->add_norm_hessian_force(norm_hessian_force[i * nforce + j]);
      }
    }
  }

  return grpc::Status::OK;
}

grpc::Status DirectService::Noise(grpc::ServerContext* context,
                                  const direct::NoiseRequest* request,
                                  direct::NoiseResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // weight
  direct::Noise input = request->noise();
  direct::Noise* output = response->mutable_noise();

  // process
  int nv = optimizer_.model->nv;
  if (input.process_size() > 0) {
    CHECK_SIZE("noise process", nv, input.process_size());
    optimizer_.noise_process.assign(input.process().begin(),
                                    input.process().end());
  }
  for (int i = 0; i < nv; i++) {
    output->add_process(optimizer_.noise_process[i]);
  }

  // sensor
  int num_sensor = optimizer_.NumberSensors();
  if (input.sensor_size() > 0) {
    CHECK_SIZE("noise sensor", num_sensor, input.sensor_size());
    optimizer_.noise_sensor.assign(input.sensor().begin(),
                                   input.sensor().end());
  }
  for (int i = 0; i < num_sensor; i++) {
    output->add_sensor(optimizer_.noise_sensor[i]);
  }

  // parameters
  int num_parameters = optimizer_.NumberParameters();
  if (num_parameters > 0) {
    if (input.parameter_size() > 0) {
      CHECK_SIZE("noise parameter", num_parameters, input.parameter_size());
      optimizer_.noise_parameter.assign(input.parameter().begin(),
                                        input.parameter().end());
    }
    for (int i = 0; i < num_parameters; i++) {
      output->add_parameter(optimizer_.noise_parameter[i]);
    }
  }

  return grpc::Status::OK;
}

grpc::Status DirectService::Reset(grpc::ServerContext* context,
                                  const direct::ResetRequest* request,
                                  direct::ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // reset
  optimizer_.Reset();

  return grpc::Status::OK;
}

grpc::Status DirectService::Optimize(grpc::ServerContext* context,
                                     const direct::OptimizeRequest* request,
                                     direct::OptimizeResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // optimize
  optimizer_.Optimize();

  return grpc::Status::OK;
}

grpc::Status DirectService::Status(grpc::ServerContext* context,
                                   const direct::StatusRequest* request,
                                   direct::StatusResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // status
  direct::Status* status = response->mutable_status();

  // search iterations
  status->set_search_iterations(optimizer_.IterationsSearch());

  // smoother iterations
  status->set_smoother_iterations(optimizer_.IterationsSmoother());

  // step size
  status->set_step_size(optimizer_.StepSize());

  // regularization
  status->set_regularization(optimizer_.Regularization());

  // gradient norm
  status->set_gradient_norm(optimizer_.GradientNorm());

  // search direction norm
  status->set_search_direction_norm(optimizer_.SearchDirectionNorm());

  // solve status
  status->set_solve_status(static_cast<int>(optimizer_.SolveStatus()));

  // cost difference
  status->set_cost_difference(optimizer_.CostDifference());

  // improvement
  status->set_improvement(optimizer_.Improvement());

  // expected
  status->set_expected(optimizer_.Expected());

  // reduction ratio
  status->set_reduction_ratio(optimizer_.ReductionRatio());

  return grpc::Status::OK;
}

grpc::Status DirectService::SensorInfo(grpc::ServerContext* context,
                                       const direct::SensorInfoRequest* request,
                                       direct::SensorInfoResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }

  // start index
  response->set_start_index(optimizer_.SensorStartIndex());

  // number of sensor measurements
  response->set_num_measurements(optimizer_.NumberSensors());

  // sensor measurement dimension
  response->set_dim_measurements(optimizer_.DimensionSensor());

  return grpc::Status::OK;
}

#undef CHECK_SIZE

}  // namespace mjpc::direct_grpc
