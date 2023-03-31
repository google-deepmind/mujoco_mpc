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

// Unit tests for the `AgentServiceImpl` class.

#include "grpc/agent_service_impl.h"

#include <memory>
#include <string_view>

#include "testing/base/public/gunit.h"
#include <grpcpp/channel.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/channel_arguments.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>
#include "grpc/agent.grpc.pb.h"
#include "grpc/agent.pb.h"

namespace agent_grpc {

using agent::grpc_gen::Agent;

class AgentServiceImplTest : public ::testing::Test {
 protected:
  void SetUp() override {
    agent_service_impl = std::make_unique<AgentServiceImpl>();
    grpc::ServerBuilder builder;
    builder.RegisterService(agent_service_impl.get());
    server = builder.BuildAndStart();
    std::shared_ptr<grpc::Channel> channel =
        server->InProcessChannel(grpc::ChannelArguments());
    stub = Agent::NewStub(channel);
  }

  void TearDown() override { server->Shutdown(); }

  void RunAndCheckInit(std::string_view task_id, mjModel* model) {
    grpc::ClientContext init_context;

    agent::InitRequest init_request;
    init_request.set_task_id(task_id);

    if (!model) {
      // TODO(khartikainen): test case where we pass in the model. We already do
      // this on the python side but should check here as well.
      ASSERT_FALSE(model);
    } else {
      init_request.set_allocated_model(nullptr);
    }

    agent::InitResponse init_response;
    grpc::Status init_status =
        stub->Init(&init_context, init_request, &init_response);

    EXPECT_TRUE(init_status.ok()) << init_status.error_message();
  }

  std::unique_ptr<AgentServiceImpl> agent_service_impl;
  std::unique_ptr<Agent::Stub> stub;
  std::unique_ptr<grpc::Server> server;
};

TEST_F(AgentServiceImplTest, Init_WithoutModel) {
  RunAndCheckInit("Cartpole", NULL);
}

TEST_F(AgentServiceImplTest, Init_WithModel) {
  // TODO(khartikainen)
}

TEST_F(AgentServiceImplTest, SetState_Works) {
  RunAndCheckInit("Cartpole", NULL);

  grpc::ClientContext set_state_context;

  agent::SetStateRequest set_state_request;
  agent::State* state = new agent::State();
  state->set_time(0.0);
  set_state_request.set_allocated_state(state);
  agent::SetStateResponse set_state_response;
  grpc::Status set_state_status = stub->SetState(
      &set_state_context, set_state_request, &set_state_response);

  EXPECT_TRUE(set_state_status.ok()) << set_state_status.error_message();
}

TEST_F(AgentServiceImplTest, SetState_WrongSize) {
  RunAndCheckInit("Cartpole", NULL);

  grpc::ClientContext set_state_context;

  agent::SetStateRequest set_state_request;
  agent::State* state = new agent::State();
  state->set_time(0.0);
  state->add_qpos(0.0);
  state->add_qpos(0.0);
  state->add_qpos(0.0);
  set_state_request.set_allocated_state(state);
  agent::SetStateResponse set_state_response;
  grpc::Status set_state_status = stub->SetState(
      &set_state_context, set_state_request, &set_state_response);

  EXPECT_FALSE(set_state_status.ok());
}

TEST_F(AgentServiceImplTest, PlannerStep_ProducesNonzeroAction) {
  RunAndCheckInit("Cartpole", NULL);

  grpc::ClientContext set_task_parameter_context;

  agent::SetTaskParameterRequest set_task_parameter_request;
  set_task_parameter_request.set_name("Goal");
  set_task_parameter_request.set_value(-1.0);
  agent::SetTaskParameterResponse set_task_parameter_response;
  grpc::Status set_task_parameter_status = stub->SetTaskParameter(
      &set_task_parameter_context, set_task_parameter_request,
      &set_task_parameter_response);

  EXPECT_TRUE(set_task_parameter_status.ok());

  grpc::ClientContext planner_step_context;

  agent::PlannerStepRequest planner_step_request;
  agent::PlannerStepResponse planner_step_response;
  grpc::Status planner_step_status = stub->PlannerStep(
      &planner_step_context, planner_step_request, &planner_step_response);

  EXPECT_TRUE(planner_step_status.ok()) << planner_step_status.error_message();

  grpc::ClientContext get_action_context;

  agent::GetActionRequest get_action_request;
  agent::GetActionResponse get_action_response;
  grpc::Status get_action_status = stub->GetAction(
      &get_action_context, get_action_request, &get_action_response);

  EXPECT_TRUE(get_action_status.ok()) << get_action_status.error_message();
  EXPECT_EQ(get_action_response.action().size(), 1);
  EXPECT_TRUE(get_action_response.action()[0] != 0.0);
}

TEST_F(AgentServiceImplTest, SetTaskParameter_Works) {
  RunAndCheckInit("Cartpole", NULL);

  grpc::ClientContext set_task_parameter_context;

  agent::SetTaskParameterRequest set_task_parameter_request;
  set_task_parameter_request.set_name("Goal");
  set_task_parameter_request.set_value(16.0);
  agent::SetTaskParameterResponse set_task_parameter_response;
  grpc::Status set_task_parameter_status = stub->SetTaskParameter(
      &set_task_parameter_context, set_task_parameter_request,
      &set_task_parameter_response);

  EXPECT_TRUE(set_task_parameter_status.ok());
}

}  // namespace agent_grpc
