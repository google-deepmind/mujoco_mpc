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

// An implementation of the `Estimator` gRPC service.

#ifndef GRPC_ESTIMATOR_SERVICE_H
#define GRPC_ESTIMATOR_SERVICE_H

#include <memory>
#include <vector>

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>
#include "grpc/estimator.grpc.pb.h"
#include "grpc/estimator.pb.h"
#include "mjpc/estimators/buffer.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace estimator_grpc {

class EstimatorService final : public estimator::Estimator::Service {
 public:
  explicit EstimatorService()
      : thread_pool_(mjpc::NumAvailableHardwareThreads()) {}
  ~EstimatorService();

  grpc::Status Init(grpc::ServerContext* context,
                    const estimator::InitRequest* request,
                    estimator::InitResponse* response) override;

  grpc::Status Data(grpc::ServerContext* context,
                    const estimator::DataRequest* request,
                    estimator::DataResponse* response) override;

  grpc::Status Settings(grpc::ServerContext* context,
                        const estimator::SettingsRequest* request,
                        estimator::SettingsResponse* response) override;

  grpc::Status Cost(grpc::ServerContext* context,
                    const estimator::CostRequest* request,
                    estimator::CostResponse* response) override;

  grpc::Status Weights(grpc::ServerContext* context,
                       const estimator::WeightsRequest* request,
                       estimator::WeightsResponse* response) override;

  grpc::Status Norms(grpc::ServerContext* context,
                     const estimator::NormRequest* request,
                     estimator::NormResponse* response) override;

  grpc::Status Shift(grpc::ServerContext* context,
                     const estimator::ShiftRequest* request,
                     estimator::ShiftResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const estimator::ResetRequest* request,
                     estimator::ResetResponse* response) override;

  grpc::Status Optimize(grpc::ServerContext* context,
                        const estimator::OptimizeRequest* request,
                        estimator::OptimizeResponse* response) override;

  grpc::Status Status(grpc::ServerContext* context,
                      const estimator::StatusRequest* request,
                      estimator::StatusResponse* response) override;

  grpc::Status CostHessian(grpc::ServerContext* context,
                           const estimator::CostHessianRequest* request,
                           estimator::CostHessianResponse* response) override;

  grpc::Status PriorMatrix(grpc::ServerContext* context,
                           const estimator::PriorMatrixRequest* request,
                           estimator::PriorMatrixResponse* response) override;

  grpc::Status ResetBuffer(grpc::ServerContext* context,
                           const estimator::ResetBufferRequest* request,
                           estimator::ResetBufferResponse* response) override;

  grpc::Status BufferData(grpc::ServerContext* context,
                          const estimator::BufferDataRequest* request,
                          estimator::BufferDataResponse* response) override;

  grpc::Status UpdateBuffer(grpc::ServerContext* context,
                            const estimator::UpdateBufferRequest* request,
                            estimator::UpdateBufferResponse* response) override;

 private:
  bool Initialized() const {
    return estimator_.model_ && estimator_.configuration_length_ >= 3;
  }

  // estimator
  mjpc::Estimator estimator_;
  mjpc::UniqueMjModel estimator_model_override_ = {nullptr, mj_deleteModel};

  // buffer 
  mjpc::Buffer buffer_;
  
  // threadpool
  mjpc::ThreadPool thread_pool_;
};

}  // namespace estimator_grpc

#endif  // GRPC_ESTIMATOR_SERVICE_H
