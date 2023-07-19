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

// An implementation of the `KALMAN` gRPC service.

#ifndef GRPC_KALMAN_SERVICE_H
#define GRPC_KALMAN_SERVICE_H

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include <memory>
#include <vector>

#include "grpc/kalman.grpc.pb.h"
#include "grpc/kalman.pb.h"
#include "mjpc/estimators/kalman.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace kalman_grpc {

class KALMANService final : public kalman::KALMAN::Service {
 public:
  explicit KALMANService() : thread_pool_(1) {}
  ~KALMANService();

  grpc::Status Init(grpc::ServerContext* context,
                    const kalman::InitRequest* request,
                    kalman::InitResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const kalman::ResetRequest* request,
                     kalman::ResetResponse* response) override;

  grpc::Status Settings(grpc::ServerContext* context,
                        const kalman::SettingsRequest* request,
                        kalman::SettingsResponse* response) override;

  grpc::Status UpdateMeasurement(
      grpc::ServerContext* context,
      const kalman::UpdateMeasurementRequest* request,
      kalman::UpdateMeasurementResponse* response) override;

  grpc::Status UpdatePrediction(
      grpc::ServerContext* context, const kalman::UpdatePredictionRequest* request,
      kalman::UpdatePredictionResponse* response) override;

  grpc::Status Timers(grpc::ServerContext* context,
                      const kalman::TimersRequest* request,
                      kalman::TimersResponse* response) override;

  grpc::Status State(grpc::ServerContext* context,
                     const kalman::StateRequest* request,
                     kalman::StateResponse* response) override;

  grpc::Status Covariance(grpc::ServerContext* context,
                          const kalman::CovarianceRequest* request,
                          kalman::CovarianceResponse* response) override;

  grpc::Status Noise(grpc::ServerContext* context,
                     const kalman::NoiseRequest* request,
                     kalman::NoiseResponse* response) override;

 private:
  bool Initialized() const { return kalman_.model; }

  // kalman
  mjpc::Kalman kalman_;
  mjpc::UniqueMjModel kalman_model_override_ = {nullptr, mj_deleteModel};

  // threadpool
  mjpc::ThreadPool thread_pool_;
};

}  // namespace kalman_grpc

#endif  // GRPC_KALMAN_SERVICE_H
