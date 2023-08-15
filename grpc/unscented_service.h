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

// An implementation of the `Unscented` gRPC service.

#ifndef GRPC_UNSCENTED_SERVICE_H
#define GRPC_UNSCENTED_SERVICE_H

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include <memory>
#include <vector>

#include "grpc/unscented.grpc.pb.h"
#include "grpc/unscented.pb.h"
#include "mjpc/estimators/unscented.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

namespace unscented_grpc {

class UnscentedService final : public unscented::Unscented::Service {
 public:
  explicit UnscentedService() : thread_pool_(1) {}
  ~UnscentedService();

  grpc::Status Init(grpc::ServerContext* context,
                    const unscented::InitRequest* request,
                    unscented::InitResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const unscented::ResetRequest* request,
                     unscented::ResetResponse* response) override;

  grpc::Status Settings(grpc::ServerContext* context,
                        const unscented::SettingsRequest* request,
                        unscented::SettingsResponse* response) override;

  grpc::Status Update(grpc::ServerContext* context,
                      const unscented::UpdateRequest* request,
                      unscented::UpdateResponse* response) override;

  grpc::Status Timers(grpc::ServerContext* context,
                      const unscented::TimersRequest* request,
                      unscented::TimersResponse* response) override;

  grpc::Status State(grpc::ServerContext* context,
                     const unscented::StateRequest* request,
                     unscented::StateResponse* response) override;

  grpc::Status Covariance(grpc::ServerContext* context,
                          const unscented::CovarianceRequest* request,
                          unscented::CovarianceResponse* response) override;

  grpc::Status Noise(grpc::ServerContext* context,
                     const unscented::NoiseRequest* request,
                     unscented::NoiseResponse* response) override;

 private:
  bool Initialized() const { return unscented_.model; }

  // unscented
  mjpc::Unscented unscented_;
  mjpc::UniqueMjModel unscented_model_override_ = {nullptr, mj_deleteModel};

  // threadpool
  mjpc::ThreadPool thread_pool_;
};

}  // namespace unscented_grpc

#endif  // GRPC_UNSCENTED_SERVICE_H