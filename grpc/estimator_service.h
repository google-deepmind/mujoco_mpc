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

#ifndef GRPC_ESTIMATOR_SERVICE_H_
#define GRPC_ESTIMATOR_SERVICE_H_

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include <memory>
#include <vector>

#include "grpc/estimator.grpc.pb.h"
#include "grpc/estimator.pb.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace mjpc::batch_grpc {

class BatchService final : public batch::Batch::Service {
 public:
  explicit BatchService() : thread_pool_(mjpc::NumAvailableHardwareThreads()) {}
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

  grpc::Status Norms(grpc::ServerContext* context,
                     const batch::NormRequest* request,
                     batch::NormResponse* response) override;

  grpc::Status Shift(grpc::ServerContext* context,
                     const batch::ShiftRequest* request,
                     batch::ShiftResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const batch::ResetRequest* request,
                     batch::ResetResponse* response) override;

  grpc::Status Optimize(grpc::ServerContext* context,
                        const batch::OptimizeRequest* request,
                        batch::OptimizeResponse* response) override;

  grpc::Status Status(grpc::ServerContext* context,
                      const batch::StatusRequest* request,
                      batch::StatusResponse* response) override;

  grpc::Status Timing(grpc::ServerContext* context,
                      const batch::TimingRequest* request,
                      batch::TimingResponse* response) override;

  grpc::Status PriorWeights(grpc::ServerContext* context,
                            const batch::PriorWeightsRequest* request,
                            batch::PriorWeightsResponse* response) override;

 private:
  bool Initialized() const {
    return batch_.model && batch_.ConfigurationLength() >= 3;
  }

  // batch
  mjpc::Batch batch_;
  mjpc::UniqueMjModel batch_model_override_ = {nullptr, mj_deleteModel};

  // threadpool
  mjpc::ThreadPool thread_pool_;
};

}  // namespace mjpc::batch_grpc

#endif  // GRPC_ESTIMATOR_SERVICE_H_
