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

// Unit tests for the `AgentService` class.

#include "grpc/agent_service.h"

#include <memory>
#include <string_view>

#include "testing/base/public/gmock.h"
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
#include "third_party/mujoco_mpc/grpc/agent.proto.h" // TODO(etom): remove?
#include "mjpc/tasks/tasks.h"

namespace agent_grpc {

using agent::grpc_gen::Agent;

class AgentServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    agent_service = std::make_unique<AgentService>(mjpc::GetTasks());
    grpc::ServerBuilder builder;
    builder.RegisterService(agent_service.get());
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

  std::unique_ptr<AgentService> agent_service;
  std::unique_ptr<Agent::Stub> stub;
  std::unique_ptr<grpc::Server> server;
};

TEST_F(AgentServiceTest, Init_WithoutModel) {
  RunAndCheckInit("Cartpole", nullptr);
}

TEST_F(AgentServiceTest, Init_WithModel) {
  // TODO(khartikainen)
}

TEST_F(AgentServiceTest, SetState_Works) {
  RunAndCheckInit("Cartpole", nullptr);

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

TEST_F(AgentServiceTest, SetState_WrongSize) {
  RunAndCheckInit("Cartpole", nullptr);

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

TEST_F(AgentServiceTest, PlannerStep_ProducesNonzeroAction) {
  RunAndCheckInit("Cartpole", nullptr);

  {
    grpc::ClientContext context;
    agent::SetTaskParametersRequest request;
    (*request.mutable_parameters())["Goal"].set_numeric(-1.0);
    agent::SetTaskParametersResponse response;
    grpc::Status status = stub->SetTaskParameters(&context, request, &response);

    EXPECT_TRUE(status.ok());
  }

  {
    grpc::ClientContext context;

    agent::PlannerStepRequest request;
    agent::PlannerStepResponse response;
    grpc::Status status = stub->PlannerStep(&context, request, &response);

    EXPECT_TRUE(status.ok()) << status.error_message();
  }

  {
    grpc::ClientContext context;

    agent::GetActionRequest request;
    agent::GetActionResponse response;
    grpc::Status status = stub->GetAction(&context, request, &response);

    EXPECT_TRUE(status.ok()) << status.error_message();
    EXPECT_EQ(response.action().size(), 1);
    EXPECT_TRUE(response.action()[0] != 0.0);
  }
}

TEST_F(AgentServiceTest, Step_AdvancesTime) {
  RunAndCheckInit("Cartpole", nullptr);

  agent::State initial_state;
  {
    grpc::ClientContext context;
    agent::GetStateRequest request;
    agent::GetStateResponse response;
    EXPECT_TRUE(stub->GetState(&context, request, &response).ok());
    initial_state = response.state();
  }

  {
    grpc::ClientContext context;
    agent::SetTaskParametersRequest request;
    (*request.mutable_parameters())["Goal"].set_numeric(-1.0);
    agent::SetTaskParametersResponse response;
    EXPECT_TRUE(stub->SetTaskParameters(&context, request, &response).ok());
  }

  {
    grpc::ClientContext context;
    agent::PlannerStepRequest request;
    agent::PlannerStepResponse response;
    grpc::Status status = stub->PlannerStep(&context, request, &response);

    EXPECT_TRUE(status.ok()) << status.error_message();
  }

  for (int i = 0; i < 3; i++) {
    grpc::ClientContext context;
    agent::StepRequest request;
    agent::StepResponse response;
    grpc::Status status = stub->Step(&context, request, &response);

    EXPECT_TRUE(status.ok()) << status.error_message();
  }

  agent::State final_state;
  {
    grpc::ClientContext context;
    agent::GetStateRequest request;
    agent::GetStateResponse response;
    grpc::Status status = stub->GetState(&context, request, &response);
    EXPECT_TRUE(status.ok()) << status.error_message();
    final_state = response.state();
  }
  double cartpole_timestep = 0.001;
  EXPECT_DOUBLE_EQ(final_state.time() - initial_state.time(),
                   3 * cartpole_timestep);
}

TEST_F(AgentServiceTest, Step_CallsTransition) {
  // the goal position changes on every timestep in the Particle task, but only
  // if Transition is called.

  RunAndCheckInit("Particle", nullptr);

  agent::State initial_state;
  {
    grpc::ClientContext context;
    agent::GetStateRequest request;
    agent::GetStateResponse response;
    grpc::Status status = stub->GetState(&context, request, &response);
    EXPECT_TRUE(status.ok()) << status.error_message();
    initial_state = response.state();
  }

  {
    grpc::ClientContext context;
    agent::StepRequest request;
    agent::StepResponse response;
    grpc::Status status = stub->Step(&context, request, &response);

    EXPECT_TRUE(status.ok()) << status.error_message();
  }

  agent::State final_state;
  {
    grpc::ClientContext context;
    agent::GetStateRequest request;
    agent::GetStateResponse response;
    grpc::Status status = stub->GetState(&context, request, &response);
    EXPECT_TRUE(status.ok()) << status.error_message();
    final_state = response.state();
  }
  EXPECT_NE(final_state.mocap_pos()[0], initial_state.mocap_pos()[0])
      << "mocap_pos stayed constant. Was Transition called?";
}

TEST_F(AgentServiceTest, SetTaskParameters_Numeric) {
  RunAndCheckInit("Cartpole", nullptr);

  grpc::ClientContext context;

  agent::SetTaskParametersRequest request;
  (*request.mutable_parameters())["Goal"].set_numeric(16.0);
  agent::SetTaskParametersResponse response;
  grpc::Status status = stub->SetTaskParameters(&context, request, &response);

  EXPECT_TRUE(status.ok());
}

TEST_F(AgentServiceTest, SetTaskParameters_Select) {
  RunAndCheckInit("Quadruped Flat", nullptr);

  grpc::ClientContext context;

  agent::SetTaskParametersRequest request;
  (*request.mutable_parameters())["Gait"].set_selection("Trot");
  agent::SetTaskParametersResponse response;
  grpc::Status status = stub->SetTaskParameters(&context, request, &response);

  EXPECT_TRUE(status.ok()) << status.error_message();
}

TEST_F(AgentServiceTest, SetCostWeights_Works) {
  RunAndCheckInit("Cartpole", nullptr);

  grpc::ClientContext context;

  agent::SetCostWeightsRequest request;
  (*request.mutable_cost_weights())["Vertical"] = 99;
  (*request.mutable_cost_weights())["Velocity"] = 3;
  agent::SetCostWeightsResponse response;
  grpc::Status status = stub->SetCostWeights(&context, request, &response);

  EXPECT_TRUE(status.ok());
}

TEST_F(AgentServiceTest, SetCostWeights_RejectsInvalidName) {
  RunAndCheckInit("Cartpole", nullptr);

  grpc::ClientContext context;

  agent::SetCostWeightsRequest request;
  (*request.mutable_cost_weights())["Vertically"] = 99;
  agent::SetCostWeightsResponse response;
  grpc::Status status = stub->SetCostWeights(&context, request, &response);

  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.error_code(), grpc::StatusCode::INVALID_ARGUMENT);
  EXPECT_THAT(status.error_message(), testing::ContainsRegex("Velocity"))
      << "Error message should contain the list of cost term names.";
}

}  // namespace agent_grpc
