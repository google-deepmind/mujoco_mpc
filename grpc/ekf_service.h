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

// An implementation of the `EKF` gRPC service.

#ifndef GRPC_EKF_SERVICE_H
#define GRPC_EKF_SERVICE_H

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include <memory>
#include <vector>

#include "grpc/ekf.grpc.pb.h"
#include "grpc/ekf.pb.h"
#include "mjpc/estimators/kalman.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace ekf_grpc {

class EKFService final : public ekf::EKF::Service {
 public:
  explicit EKFService() : thread_pool_(1) {}
  ~EKFService();

  grpc::Status Init(grpc::ServerContext* context,
                    const ekf::InitRequest* request,
                    ekf::InitResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const ekf::ResetRequest* request,
                     ekf::ResetResponse* response) override;

  grpc::Status Settings(grpc::ServerContext* context,
                        const ekf::SettingsRequest* request,
                        ekf::SettingsResponse* response) override;

  grpc::Status UpdateMeasurement(
      grpc::ServerContext* context,
      const ekf::UpdateMeasurementRequest* request,
      ekf::UpdateMeasurementResponse* response) override;

  grpc::Status UpdatePrediction(
      grpc::ServerContext* context, const ekf::UpdatePredictionRequest* request,
      ekf::UpdatePredictionResponse* response) override;

  grpc::Status Timers(grpc::ServerContext* context,
                      const ekf::TimersRequest* request,
                      ekf::TimersResponse* response) override;

  grpc::Status State(grpc::ServerContext* context,
                     const ekf::StateRequest* request,
                     ekf::StateResponse* response) override;

  grpc::Status Covariance(grpc::ServerContext* context,
                          const ekf::CovarianceRequest* request,
                          ekf::CovarianceResponse* response) override;

  grpc::Status Noise(grpc::ServerContext* context,
                     const ekf::NoiseRequest* request,
                     ekf::NoiseResponse* response) override;

 private:
  bool Initialized() const { return ekf_.model; }

  // ekf
  mjpc::EKF ekf_;
  mjpc::UniqueMjModel ekf_model_override_ = {nullptr, mj_deleteModel};

  // threadpool
  mjpc::ThreadPool thread_pool_;
};

}  // namespace ekf_grpc

#endif  // GRPC_EKF_SERVICE_H
