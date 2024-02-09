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

#include <memory>
#include <string_view>

#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/channel_arguments.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "mjpc/grpc/agent_service.h"
#include "mjpc/grpc/agent.grpc.pb.h"
#include "mjpc/grpc/agent.pb.h"
#include "mjpc/grpc/agent.proto.h"
#include "mjpc/tasks/tasks.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"

namespace mjpc::agent_grpc {

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
    agent::InitRequest init_request;
    init_request.set_task_id(task_id);

    if (!model) {
      // TODO(khartikainen): test case where we pass in the model. We already do
      // this on the python side but should check here as well.
      ASSERT_FALSE(model);
    } else {
      init_request.set_allocated_model(nullptr);
    }

    SendRequest(&Agent::Stub::Init, init_request);
  }

  // Sends a request, validates the status code and returns the response
  template <class Req, class Res>
  Res SendRequest(grpc::Status (Agent::Stub::*method)(grpc::ClientContext*,
                                                      const Req&, Res*),
                  const Req& request) {
    grpc::ClientContext context;
    Res response;
    grpc::Status status = (stub.get()->*method)(&context, request, &response);
    EXPECT_TRUE(status.ok()) << status.error_message();
    return response;
  }

  // an overload which constructs an empty request
  template <class Req, class Res>
  Res SendRequest(grpc::Status (Agent::Stub::*method)(grpc::ClientContext*,
                                                      const Req&, Res*)) {
    return SendRequest(method, Req());
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

  agent::SetStateRequest set_state_request;
  agent::State* state = new agent::State();
  state->set_time(0.0);
  set_state_request.set_allocated_state(state);
  SendRequest(&Agent::Stub::SetState, set_state_request);
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
    agent::SetTaskParametersRequest request;
    (*request.mutable_parameters())["Goal"].set_numeric(-1.0);
    SendRequest(&Agent::Stub::SetTaskParameters, request);
  }

  SendRequest(&Agent::Stub::PlannerStep);

  {
    agent::GetActionResponse response = SendRequest(&Agent::Stub::GetAction);

    ASSERT_EQ(response.action().size(), 1);
    EXPECT_TRUE(response.action()[0] != 0.0);
  }
}

TEST_F(AgentServiceTest, ActionAveragingGivesDifferentResult) {
  RunAndCheckInit("Cartpole", nullptr);

  {
    agent::SetTaskParametersRequest request;
    (*request.mutable_parameters())["Goal"].set_numeric(-1.0);
    SendRequest(&Agent::Stub::SetTaskParameters, request);
  }

  SendRequest(&Agent::Stub::PlannerStep);

  double action_without_averaging;
  {
    agent::GetActionResponse response = SendRequest(&Agent::Stub::GetAction);
    ASSERT_EQ(response.action().size(), 1);
    EXPECT_TRUE(response.action()[0] != 0.0);
    action_without_averaging = response.action()[0];
  }

  double action_with_averaging;
  {
    grpc::ClientContext context;

    agent::GetActionRequest request;
    request.set_averaging_duration(1.0);
    agent::GetActionResponse response =
        SendRequest(&Agent::Stub::GetAction, request);
    ASSERT_EQ(response.action().size(), 1);
    EXPECT_TRUE(response.action()[0] != 0.0);
    action_with_averaging = response.action()[0];
  }
  EXPECT_NE(action_with_averaging, action_without_averaging);
}

TEST_F(AgentServiceTest, NominalActionIndependentOfState) {
  // Pick a task that uses iLQG, where there is normally a feedback term on the
  // policy.
  RunAndCheckInit("Swimmer", nullptr);

  SendRequest(&Agent::Stub::PlannerStep);

  double nominal_action1;
  {
    agent::GetActionRequest request;
    request.set_averaging_duration(1.0);
    request.set_nominal_action(true);
    request.set_time(0.01);
    agent::GetActionResponse response =
        SendRequest(&Agent::Stub::GetAction, request);
    EXPECT_GE(response.action().size(), 1);
    nominal_action1 = response.action()[0];
    EXPECT_NE(nominal_action1, 0.0);
  }

  // Set a new state
  {
    agent::SetStateRequest request;
    static constexpr int kSwimmerDofs = 8;
    for (int i = 0; i < kSwimmerDofs; i++) {
      request.mutable_state()->mutable_qpos()->Add(0.1);
    }
    SendRequest(&Agent::Stub::SetState, request);
  }

  double nominal_action2;
  {
    agent::GetActionRequest request;
    request.set_averaging_duration(1.0);
    request.set_nominal_action(true);
    request.set_time(0.01);
    agent::GetActionResponse response =
        SendRequest(&Agent::Stub::GetAction, request);
    EXPECT_GE(response.action().size(), 1);
    nominal_action2 = response.action()[0];
  }

  double feedback_action;
  {
    agent::GetActionRequest request;
    request.set_averaging_duration(1.0);
    request.set_nominal_action(false);
    request.set_time(0.01);
    agent::GetActionResponse response =
        SendRequest(&Agent::Stub::GetAction, request);
    EXPECT_GE(response.action().size(), 1);
    feedback_action = response.action()[0];
  }

  EXPECT_EQ(nominal_action1, nominal_action2)
      << "nominal action should be the same";
  EXPECT_NE(nominal_action1, feedback_action)
      << "feedback action should be different from the nominal";
}

TEST_F(AgentServiceTest, Step_AdvancesTime) {
  RunAndCheckInit("Cartpole", nullptr);

  agent::State initial_state = SendRequest(&Agent::Stub::GetState).state();

  {
    agent::SetTaskParametersRequest request;
    (*request.mutable_parameters())["Goal"].set_numeric(-1.0);
    SendRequest(&Agent::Stub::SetTaskParameters, request);
  }

  {
    grpc::ClientContext context;
    agent::GetTaskParametersRequest request;
    agent::GetTaskParametersResponse response;
    grpc::Status status = stub->GetTaskParameters(&context, request, &response);
    EXPECT_TRUE(status.ok());
    EXPECT_EQ(response.parameters().at("Goal").numeric(), -1.0);
  }

  SendRequest(&Agent::Stub::PlannerStep);
  for (int i = 0; i < 3; i++) {
    SendRequest(&Agent::Stub::Step);
  }

  agent::State final_state = SendRequest(&Agent::Stub::GetState).state();
  double cartpole_timestep = 0.001;
  EXPECT_DOUBLE_EQ(final_state.time() - initial_state.time(),
                   3 * cartpole_timestep);
}

TEST_F(AgentServiceTest, Step_CallsTransition) {
  // the goal position changes on every timestep in the Particle task, but only
  // if Transition is called.

  RunAndCheckInit("Particle", nullptr);

  agent::State initial_state = SendRequest(&Agent::Stub::GetState).state();

  SendRequest(&Agent::Stub::Step);

  agent::State final_state = SendRequest(&Agent::Stub::GetState).state();
  EXPECT_NE(final_state.mocap_pos()[0], initial_state.mocap_pos()[0])
      << "mocap_pos stayed constant. Was Transition called?";
}

TEST_F(AgentServiceTest, SetTaskParameters_Numeric) {
  RunAndCheckInit("Cartpole", nullptr);

  agent::SetTaskParametersRequest request;
  (*request.mutable_parameters())["Goal"].set_numeric(16.0);
  SendRequest(&Agent::Stub::SetTaskParameters, request);
}

TEST_F(AgentServiceTest, SetTaskParameters_Select) {
  RunAndCheckInit("Quadruped Flat", nullptr);
  agent::SetTaskParametersRequest request;
  (*request.mutable_parameters())["Gait"].set_selection("Trot");
  SendRequest(&Agent::Stub::SetTaskParameters, request);
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

TEST_F(AgentServiceTest, GetMode_Works) {
  RunAndCheckInit("Cartpole", nullptr);

  grpc::ClientContext context;

  agent::GetModeRequest request;
  agent::GetModeResponse response;
  grpc::Status status = stub->GetMode(&context, request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(response.mode(), "default_mode");
}

TEST_F(AgentServiceTest, GetAllModes_Works) {
  RunAndCheckInit("Cartpole", nullptr);

  grpc::ClientContext context;

  agent::GetAllModesRequest request;
  agent::GetAllModesResponse response;
  grpc::Status status = stub->GetAllModes(&context, request, &response);

  EXPECT_TRUE(status.ok());
  EXPECT_EQ(response.mode_names().size(), 1);
  EXPECT_EQ(response.mode_names()[0], "default_mode");
}

TEST_F(AgentServiceTest, GetResiduals_Works) {
  RunAndCheckInit("Cartpole", nullptr);

  grpc::ClientContext context;

  agent::GetResidualsRequest request;
  agent::GetResidualsResponse response;
  grpc::Status status = stub->GetResiduals(&context, request, &response);

  EXPECT_TRUE(status.ok());
}

}  // namespace mjpc::agent_grpc
