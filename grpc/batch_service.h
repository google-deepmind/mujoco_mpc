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

// An implementation of the `Batch` gRPC service.

#ifndef GRPC_BATCH_SERVICE_H_
#define GRPC_BATCH_SERVICE_H_

#include <memory>
#include <vector>

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "grpc/batch.grpc.pb.h"
#include "grpc/batch.pb.h"
#include "mjpc/direct/direct.h"
#include "mjpc/utilities.h"

namespace mjpc::batch_grpc {

class BatchService final : public batch::Batch::Service {
 public:
  explicit BatchService() {}
  ~BatchService();

  grpc::Status Init(grpc::ServerContext* context,
                    const batch::InitRequest* request,
                    batch::InitResponse* response) override;

  grpc::Status Data(grpc::ServerContext* context,
                    const batch::DataRequest* request,
                    batch::DataResponse* response) override;

  grpc::Status Settings(grpc::ServerContext* context,
                        const batch::SettingsRequest* request,
                        batch::SettingsResponse* response) override;

  grpc::Status Cost(grpc::ServerContext* context,
                    const batch::CostRequest* request,
                    batch::CostResponse* response) override;

  grpc::Status Noise(grpc::ServerContext* context,
                     const batch::NoiseRequest* request,
                     batch::NoiseResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const batch::ResetRequest* request,
                     batch::ResetResponse* response) override;

  grpc::Status Optimize(grpc::ServerContext* context,
                        const batch::OptimizeRequest* request,
                        batch::OptimizeResponse* response) override;

  grpc::Status Status(grpc::ServerContext* context,
                      const batch::StatusRequest* request,
                      batch::StatusResponse* response) override;

  grpc::Status SensorInfo(grpc::ServerContext* context,
                          const batch::SensorInfoRequest* request,
                          batch::SensorInfoResponse* response) override;

 private:
  bool Initialized() const {
    return optimizer_.model && optimizer_.ConfigurationLength() >= 3;
  }

  // direct optimizer
  mjpc::Direct optimizer_;
  mjpc::UniqueMjModel model_override_ = {nullptr, mj_deleteModel};
};

}  // namespace mjpc::batch_grpc

#endif  // GRPC_BATCH_SERVICE_H_
