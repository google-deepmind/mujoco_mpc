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

#include "grpc/agent_service.h"

#include <string_view>
#include <vector>

#include <absl/log/check.h>
#include <absl/strings/str_format.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>
#include "grpc/agent.pb.h"
#include "grpc/grpc_agent_util.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/task.h"

namespace agent_grpc {

using ::agent::GetActionRequest;
using ::agent::GetActionResponse;
using ::agent::GetCostValuesAndWeightsRequest;
using ::agent::GetCostValuesAndWeightsResponse;
using ::agent::GetStateRequest;
using ::agent::GetStateResponse;
using ::agent::InitRequest;
using ::agent::InitResponse;
using ::agent::PlannerStepRequest;
using ::agent::PlannerStepResponse;
using ::agent::ResetRequest;
using ::agent::ResetResponse;
using ::agent::SetCostWeightsRequest;
using ::agent::SetCostWeightsResponse;
using ::agent::SetModeRequest;
using ::agent::SetModeResponse;
using ::agent::GetModeRequest;
using ::agent::GetModeResponse;
using ::agent::SetStateRequest;
using ::agent::SetStateResponse;
using ::agent::SetTaskParametersRequest;
using ::agent::SetTaskParametersResponse;
using ::agent::StepRequest;
using ::agent::StepResponse;
using ::agent::InitEstimatorRequest;
using ::agent::InitEstimatorResponse;
using ::agent::SetEstimatorDataRequest;
using ::agent::SetEstimatorDataResponse;
using ::agent::GetEstimatorDataRequest;
using ::agent::GetEstimatorDataResponse;
using ::agent::SetEstimatorSettingsRequest;
using ::agent::SetEstimatorSettingsResponse;
using ::agent::GetEstimatorSettingsRequest;
using ::agent::GetEstimatorSettingsResponse;
using ::agent::GetEstimatorCostsRequest;
using ::agent::GetEstimatorCostsResponse;


// task used to define desired behaviour
mjpc::Task* task = nullptr;

// model used for physics
mjModel* model = nullptr;

// model used for planning, owned by the Agent instance.
mjModel* agent_model = nullptr;

void residual_sensor_callback(const mjModel* m, mjData* d, int stage) {
  // with the `m == model` guard in place, no need to clear the callback.
  if (m == agent_model || m == model) {
    if (stage == mjSTAGE_ACC) {
      task->Residual(m, d, d->sensordata);
    }
  }
}

grpc::Status AgentService::Init(grpc::ServerContext* context,
                                const InitRequest* request,
                                InitResponse* response) {
  agent_.SetTaskList(tasks_);
  grpc::Status status = grpc_agent_util::InitAgent(&agent_, request);
  if (!status.ok()) {
    return status;
  }
  agent_.SetTaskList(tasks_);
  std::string_view task_id = request->task_id();
  int task_index = agent_.GetTaskIdByName(task_id);
  if (task_index == -1) {
    return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT,
                        absl::StrFormat("Invalid task_id: '%s'", task_id));
  }
  agent_.SetTaskByIndex(task_index);

  auto load_model = agent_.LoadModel();
  agent_.Initialize(load_model.model.get());
  agent_.Allocate();
  agent_.Reset();

  task = agent_.ActiveTask();
  CHECK_EQ(agent_model, nullptr)
      << "Multiple instances of AgentService detected.";
  agent_model = agent_.GetModel();
  // copy the model before agent model's timestep and integrator are updated
  CHECK_EQ(model, nullptr)
      << "Multiple instances of AgentService detected.";
  model = mj_copyModel(nullptr, agent_model);
  data_ = mj_makeData(model);
  mjcb_sensor = residual_sensor_callback;

  agent_.SetState(data_);

  agent_.plan_enabled = true;
  agent_.action_enabled = true;

  return grpc::Status::OK;
}

AgentService::~AgentService() {
  if (data_) mj_deleteData(data_);
  if (model) mj_deleteModel(model);
  model = nullptr;
  // no need to delete agent_model and task, since they're owned by agent_.
  agent_model = nullptr;
  task = nullptr;
  mjcb_sensor = nullptr;
}

grpc::Status AgentService::GetState(grpc::ServerContext* context,
                                    const GetStateRequest* request,
                                    GetStateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  agent_.ActiveState().CopyTo(model, data_);
  return grpc_agent_util::GetState(model, data_, response);
}

grpc::Status AgentService::SetState(grpc::ServerContext* context,
                                    const SetStateRequest* request,
                                    SetStateResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  grpc::Status status =
      grpc_agent_util::SetState(request, &agent_, model, data_);
  if (!status.ok()) return status;

  mj_forward(model, data_);
  task->Transition(model, data_);

  return grpc::Status::OK;
}

grpc::Status AgentService::GetAction(grpc::ServerContext* context,
                                     const GetActionRequest* request,
                                     GetActionResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  return grpc_agent_util::GetAction(request, &agent_, response);
}

grpc::Status AgentService::GetCostValuesAndWeights(
    grpc::ServerContext* context, const GetCostValuesAndWeightsRequest* request,
    GetCostValuesAndWeightsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  return grpc_agent_util::GetCostValuesAndWeights(request, &agent_, model,
                                                  data_, response);
}

grpc::Status AgentService::PlannerStep(grpc::ServerContext* context,
                                       const PlannerStepRequest* request,
                                       PlannerStepResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  agent_.plan_enabled = true;
  agent_.PlanIteration(&thread_pool_);

  return grpc::Status::OK;
}

grpc::Status AgentService::Step(grpc::ServerContext* context,
                                const StepRequest* request,
                                StepResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  mjpc::State& state = agent_.ActiveState();
  state.CopyTo(model, data_);
  agent_.ActiveTask()->Transition(model, data_);
  agent_.ActivePlanner().ActionFromPolicy(data_->ctrl, state.state().data(),
                                          state.time(),
                                          request->use_previous_policy());
  mj_step(model, data_);
  state.Set(model, data_);
  return grpc::Status::OK;
}

grpc::Status AgentService::Reset(grpc::ServerContext* context,
                                 const ResetRequest* request,
                                 ResetResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  return grpc_agent_util::Reset(&agent_, agent_.GetModel(), data_);
}

grpc::Status AgentService::SetTaskParameters(
    grpc::ServerContext* context, const SetTaskParametersRequest* request,
    SetTaskParametersResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  return grpc_agent_util::SetTaskParameters(request, &agent_);
}

grpc::Status AgentService::SetCostWeights(
    grpc::ServerContext* context, const SetCostWeightsRequest* request,
    SetCostWeightsResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  return grpc_agent_util::SetCostWeights(request, &agent_);
}

grpc::Status AgentService::SetMode(grpc::ServerContext* context,
                                   const SetModeRequest* request,
                                   SetModeResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  return grpc_agent_util::SetMode(request, &agent_);
}

grpc::Status AgentService::GetMode(grpc::ServerContext* context,
                                   const GetModeRequest* request,
                                   GetModeResponse* response) {
  if (!Initialized()) {
    return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  }
  return grpc_agent_util::GetMode(request, &agent_, response);
}

grpc::Status AgentService::InitEstimator(
    grpc::ServerContext *context,
    const agent::InitEstimatorRequest *request,
    agent::InitEstimatorResponse *response)
{
  // if (!Initialized()) {
  //   return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  // }

  // ----- initialize with model ----- //
  mjpc::UniqueMjModel tmp_model = {nullptr, mj_deleteModel};

  // convert message
  if (request->has_model() && request->model().has_xml()) {
    std::string_view model_xml = request->model().xml();
    char load_error[1024] = "";
    tmp_model = grpc_agent_util::LoadModelFromString(model_xml, load_error, sizeof(load_error));
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

grpc::Status AgentService::SetEstimatorData(
    grpc::ServerContext *context,
    const agent::SetEstimatorDataRequest *request,
    agent::SetEstimatorDataResponse *response)
{
  // if (!Initialized()) {
  //   return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  // }

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

grpc::Status AgentService::GetEstimatorData(
    grpc::ServerContext* context, const agent::GetEstimatorDataRequest* request,
    agent::GetEstimatorDataResponse* response) {
  // if (!Initialized()) {
  //   return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  // }

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

grpc::Status AgentService::SetEstimatorSettings(
    grpc::ServerContext *context,
    const agent::SetEstimatorSettingsRequest *request,
    agent::SetEstimatorSettingsResponse *response)
{
  // if (!Initialized()) {
  //   return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  // }

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

grpc::Status AgentService::GetEstimatorSettings(
    grpc::ServerContext* context, const agent::GetEstimatorSettingsRequest* request,
    agent::GetEstimatorSettingsResponse* response) {
  // if (!Initialized()) {
  //   return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  // }

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

grpc::Status AgentService::GetEstimatorCosts(
    grpc::ServerContext* context, const agent::GetEstimatorCostsRequest* request,
    agent::GetEstimatorCostsResponse* response) {
  // if (!Initialized()) {
  //   return {grpc::StatusCode::FAILED_PRECONDITION, "Init not called."};
  // }

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

}  // namespace agent_grpc
