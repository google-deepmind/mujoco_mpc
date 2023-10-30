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

// An implementation of the `Filter` gRPC service.

#ifndef MJPC_MJPC_GRPC_FILTER_SERVICE_H_
#define MJPC_MJPC_GRPC_FILTER_SERVICE_H_

#include <memory>
#include <vector>

#include <grpcpp/server_context.h>
#include <grpcpp/support/status.h>
#include <mujoco/mujoco.h>

#include "mjpc/grpc/filter.grpc.pb.h"
#include "mjpc/grpc/filter.pb.h"
#include "mjpc/estimators/include.h"

namespace filter_grpc {

class FilterService final : public filter::StateEstimation::Service {
 public:
  explicit FilterService() : filters_(mjpc::LoadEstimators()) {}
  ~FilterService();

  grpc::Status Init(grpc::ServerContext* context,
                    const filter::InitRequest* request,
                    filter::InitResponse* response) override;

  grpc::Status Reset(grpc::ServerContext* context,
                     const filter::ResetRequest* request,
                     filter::ResetResponse* response) override;

  grpc::Status Update(grpc::ServerContext* context,
                      const filter::UpdateRequest* request,
                      filter::UpdateResponse* response) override;

  grpc::Status State(grpc::ServerContext* context,
                     const filter::StateRequest* request,
                     filter::StateResponse* response) override;

  grpc::Status Covariance(grpc::ServerContext* context,
                          const filter::CovarianceRequest* request,
                          filter::CovarianceResponse* response) override;

  grpc::Status Noise(grpc::ServerContext* context,
                     const filter::NoiseRequest* request,
                     filter::NoiseResponse* response) override;

 private:
  bool Initialized() const { return filters_[filter_]->Model(); }

  // filters
  std::vector<std::unique_ptr<mjpc::Estimator>> filters_;
  int filter_;

  // model
  mjpc::UniqueMjModel model_override_ = {nullptr, mj_deleteModel};
};

}  // namespace filter_grpc

#endif  // MJPC_MJPC_GRPC_FILTER_SERVICE_H_
