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

// A version of the MJPC UI that serves the AgentService API.

#include <cstdint>
#include <memory>
#include <string>

#include <absl/flags/parse.h>
#include <absl/flags/flag.h>
#include <absl/log/log.h>
#include <absl/strings/str_cat.h>
#include <absl/time/clock.h>
#include <absl/time/time.h>
// DEEPMIND INTERNAL IMPORT
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>

#include "mjpc/app.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/grpc/ui_agent_service.h"

ABSL_FLAG(int32_t, mjpc_port, 10000, "port to listen on");
ABSL_FLAG(int32_t, mjpc_workers, -1,
          "Not used. Here for compatibility with agent_server.cc");

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  int port = absl::GetFlag(FLAGS_mjpc_port);

  std::string server_address = absl::StrCat("[::]:", port);

  std::shared_ptr<grpc::ServerCredentials> server_credentials =
      grpc::experimental::LocalServerCredentials(LOCAL_TCP);
  grpc::ServerBuilder builder;
  builder.AddListeningPort(server_address, server_credentials);

  mjpc::MjpcApp app(mjpc::GetTasks());
  mjpc::agent_grpc::UiAgentService service(app.Sim());
  builder.SetMaxReceiveMessageSize(80 * 1024 * 1024);
  builder.RegisterService(&service);

  std::unique_ptr<grpc::Server> server(builder.BuildAndStart());
  LOG(INFO) << "Server listening on " << server_address;

  app.Start();
  server->Shutdown(absl::ToChronoTime(absl::Now() + absl::Seconds(1)));
  server->Wait();

  return 0;
}
