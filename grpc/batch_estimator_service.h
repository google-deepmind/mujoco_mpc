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

// An implementation of the `BatchEstimator` gRPC service.

#ifndef GRPC_BATCH_ESTIMATOR_SERVICE_H
#define GRPC_BATCH_ESTIMATOR_SERVICE_H

#include <memory>
#include <vector>

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>
#include "grpc/batch_estimator.grpc.pb.h"
#include "grpc/batch_estimator.pb.h"
#include "mjpc/estimators/buffer.h"
#include "mjpc/estimators/batch.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace batch_estimator_grpc {

class BatchEstimatorService final : public batch_estimator::BatchEstimator::Service {
 public:
  explicit BatchEstimatorService()
      : thread_pool_(mjpc::NumAvailableHardwareThreads()) {}
  ~BatchEstimatorService();

  grpc::Status Init(grpc::ServerContext* context,
                    const batch_estimator::InitRequest* request,
                    batch_estimator::InitResponse* response) override;

  grpc::Status Data(grpc::ServerContext* context,
                    const batch_estimator::DataRequest* request,
                    batch_estimator::DataResponse* response) override;

  grpc::Status Settings(grpc::ServerContext* context,
                        const batch_estimator::SettingsRequest* request,
                        batch_estimator::SettingsResponse* response) override;

  grpc::Status Cost(grpc::ServerContext* context,
                    const batch_estimator::CostRequest* request,
                    batch_estimator::CostResponse* response) override;

  grpc::Status Weights(grpc::ServerContext* context,
                       const batch_estimator::WeightsRequest* request,
                       batch_estimator::WeightsResponse* response) override;

  grpc::Status Norms(grpc::ServerContext* context,
                     const batch_estimator::NormRequest* request,
                     batch_estimator::NormResponse* response) override;

  grpc::Status Shift(grpc::ServerContext* context,
                     const batch_estimator::ShiftRequest* request,
                     batch_estimator::ShiftResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const batch_estimator::ResetRequest* request,
                     batch_estimator::ResetResponse* response) override;

  grpc::Status InitializeData(
      grpc::ServerContext* context,
      const batch_estimator::InitializeDataRequest* request,
      batch_estimator::InitializeDataResponse* response) override;

  grpc::Status UpdateData(grpc::ServerContext* context,
                          const batch_estimator::UpdateDataRequest* request,
                          batch_estimator::UpdateDataResponse* response) override;

  grpc::Status Optimize(grpc::ServerContext* context,
                        const batch_estimator::OptimizeRequest* request,
                        batch_estimator::OptimizeResponse* response) override;

  grpc::Status Update(grpc::ServerContext* context,
                      const batch_estimator::UpdateRequest* request,
                      batch_estimator::UpdateResponse* response) override;

  grpc::Status InitialState(grpc::ServerContext* context,
                            const batch_estimator::InitialStateRequest* request,
                            batch_estimator::InitialStateResponse* response) override;

  grpc::Status Status(grpc::ServerContext* context,
                      const batch_estimator::StatusRequest* request,
                      batch_estimator::StatusResponse* response) override;

  grpc::Status Timing(grpc::ServerContext* context,
                      const batch_estimator::TimingRequest* request,
                      batch_estimator::TimingResponse* response) override;

  grpc::Status PriorMatrix(grpc::ServerContext* context,
                           const batch_estimator::PriorMatrixRequest* request,
                           batch_estimator::PriorMatrixResponse* response) override;

  grpc::Status ResetBuffer(grpc::ServerContext* context,
                           const batch_estimator::ResetBufferRequest* request,
                           batch_estimator::ResetBufferResponse* response) override;

  grpc::Status BufferData(grpc::ServerContext* context,
                          const batch_estimator::BufferDataRequest* request,
                          batch_estimator::BufferDataResponse* response) override;

  grpc::Status UpdateBuffer(grpc::ServerContext* context,
                            const batch_estimator::UpdateBufferRequest* request,
                            batch_estimator::UpdateBufferResponse* response) override;

 private:
  bool Initialized() const {
    return batch_estimator_.model && batch_estimator_.ConfigurationLength() >= 3;
  }

  // batch_estimator
  mjpc::Batch batch_estimator_;
  mjpc::UniqueMjModel batch_estimator_model_override_ = {nullptr, mj_deleteModel};

  // buffer
  mjpc::Buffer buffer_;

  // threadpool
  mjpc::ThreadPool thread_pool_;
};

}  // namespace batch_estimator_grpc

#endif  // GRPC_BATCH_ESTIMATOR_SERVICE_H
