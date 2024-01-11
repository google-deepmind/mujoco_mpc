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

// An implementation of the `Direct` gRPC service.

#ifndef MJPC_MJPC_GRPC_DIRECT_SERVICE_H_
#define MJPC_MJPC_GRPC_DIRECT_SERVICE_H_

#include <memory>
#include <vector>

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "mjpc/grpc/direct.grpc.pb.h"
#include "mjpc/grpc/direct.pb.h"
#include "mjpc/direct/direct.h"
#include "mjpc/utilities.h"

namespace mjpc::direct_grpc {

class DirectService final : public direct::Direct::Service {
 public:
  explicit DirectService() {}
  ~DirectService();

  grpc::Status Init(grpc::ServerContext* context,
                    const direct::InitRequest* request,
                    direct::InitResponse* response) override;

  grpc::Status Data(grpc::ServerContext* context,
                    const direct::DataRequest* request,
                    direct::DataResponse* response) override;

  grpc::Status Settings(grpc::ServerContext* context,
                        const direct::SettingsRequest* request,
                        direct::SettingsResponse* response) override;

  grpc::Status Cost(grpc::ServerContext* context,
                    const direct::CostRequest* request,
                    direct::CostResponse* response) override;

  grpc::Status Noise(grpc::ServerContext* context,
                     const direct::NoiseRequest* request,
                     direct::NoiseResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const direct::ResetRequest* request,
                     direct::ResetResponse* response) override;

  grpc::Status Optimize(grpc::ServerContext* context,
                        const direct::OptimizeRequest* request,
                        direct::OptimizeResponse* response) override;

  grpc::Status Status(grpc::ServerContext* context,
                      const direct::StatusRequest* request,
                      direct::StatusResponse* response) override;

  grpc::Status SensorInfo(grpc::ServerContext* context,
                          const direct::SensorInfoRequest* request,
                          direct::SensorInfoResponse* response) override;

 private:
  bool Initialized() const {
    return optimizer_.model && optimizer_.ConfigurationLength() >= 3;
  }

  // direct optimizer
  mjpc::Direct optimizer_;
  mjpc::UniqueMjModel model_override_ = {nullptr, mj_deleteModel};
};

}  // namespace mjpc::direct_grpc

#endif  // MJPC_MJPC_GRPC_DIRECT_SERVICE_H_
