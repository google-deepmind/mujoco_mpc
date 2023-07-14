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

// Unit tests for the `EstimatorService` class.

#include "grpc/estimator_service.h"

#include <memory>
#include <string_view>

#include <grpcpp/channel.h>
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
#include <grpcpp/support/channel_arguments.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "grpc/estimator.grpc.pb.h"
#include "grpc/estimator.pb.h"

namespace estimator_grpc {

using estimator::grpc_gen::Estimator;

class EstimatorServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    estimator_service = std::make_unique<EstimatorService>();
    grpc::ServerBuilder builder;
    builder.RegisterService(estimator_service.get());
    server = builder.BuildAndStart();
    std::shared_ptr<grpc::Channel> channel =
        server->InProcessChannel(grpc::ChannelArguments());
    stub = Estimator::NewStub(channel);
  }

  void TearDown() override { server->Shutdown(); }

  void RunAndCheckInit() {
    grpc::ClientContext init_context;

    estimator::InitRequest init_request;

    estimator::InitResponse init_response;
    grpc::Status init_status =
        stub->Init(&init_context, init_request, &init_response);

    EXPECT_TRUE(init_status.ok()) << init_status.error_message();
  }

  std::unique_ptr<EstimatorService> estimator_service;
  std::unique_ptr<Estimator::Stub> stub;
  std::unique_ptr<grpc::Server> server;
};

}  // namespace estimator_grpc
