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

// Unit tests for the `BatchEstimatorService` class.

#include "grpc/batch_estimator_service.h"

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
#include "grpc/batch_estimator.grpc.pb.h"
#include "grpc/batch_estimator.pb.h"

namespace batch_estimator_grpc {

using batch_estimator::grpc_gen::BatchEstimator;

class BatchEstimatorServiceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    batch_estimator_service = std::make_unique<BatchEstimatorService>();
    grpc::ServerBuilder builder;
    builder.RegisterService(batch_estimator_service.get());
    server = builder.BuildAndStart();
    std::shared_ptr<grpc::Channel> channel =
        server->InProcessChannel(grpc::ChannelArguments());
    stub = BatchEstimator::NewStub(channel);
  }

  void TearDown() override { server->Shutdown(); }

  void RunAndCheckInit() {
    grpc::ClientContext init_context;

    batch_estimator::InitRequest init_request;

    batch_estimator::InitResponse init_response;
    grpc::Status init_status =
        stub->Init(&init_context, init_request, &init_response);

    EXPECT_TRUE(init_status.ok()) << init_status.error_message();
  }

  std::unique_ptr<BatchEstimatorService> batch_estimator_service;
  std::unique_ptr<BatchEstimator::Stub> stub;
  std::unique_ptr<grpc::Server> server;
};

}  // namespace batch_estimator_grpc
