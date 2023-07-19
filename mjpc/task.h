// Copyright 2022 DeepMind Technologies Limited
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

#ifndef MJPC_TASK_H_
#define MJPC_TASK_H_

#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <mujoco/mujoco.h>
#include "mjpc/norm.h"

namespace mjpc {

// tolerance for risk-neutral cost
inline constexpr double kRiskNeutralTolerance = 1.0e-6;

// maximum cost terms
inline constexpr int kMaxCostTerms = 64;

class Task;

// abstract class for a residual function
class ResidualFn {
 public:
  virtual ~ResidualFn() = default;

  virtual void Residual(const mjModel* model, const mjData* data,
                        double* residual) const = 0;
  virtual void CostTerms(double* terms, const double* residual,
                         bool weighted = true) const = 0;
  virtual double CostValue(const double* residual) const = 0;

  // copies weights and parameters from the Task instance. This should be
  // called from the Task class.
  virtual void Update() = 0;
};

// base implementation for ResidualFn implementations
class BaseResidualFn : public ResidualFn {
 public:
  explicit BaseResidualFn(const Task* task);
  virtual ~BaseResidualFn() = default;

  void CostTerms(double* terms, const double* residual,
                 bool weighted = true) const override;
  double CostValue(const double* residual) const override;
  void Update() override;

 protected:
  int num_residual_;
  int num_term_;
  int num_trace_;
  std::vector<int> dim_norm_residual_;
  std::vector<int> num_norm_parameter_;
  std::vector<NormType> norm_;
  std::vector<double> weight_;
  std::vector<double> norm_parameter_;
  double risk_;
  std::vector<double> parameters_;
  const Task* task_;
};

namespace internal {
// a ResidualFn which simply uses weights from the Task instance, for backwards
// compatibility.
// this isn't thread safe, because weights and parameters in the task can change
// at any time.
class ForwardingResidualFn : public ResidualFn {
 public:
  explicit ForwardingResidualFn(const Task* task) : task_(task) {}
  virtual ~ForwardingResidualFn() = default;

  void Residual(const mjModel* model, const mjData* data,
                double* residual) const override;

  void CostTerms(double* terms, const double* residual,
                         bool weighted = true) const override;
  double CostValue(const double* residual) const override;
  void Update() override {}

 private:
  const Task* task_;
};
}  // namespace internal

class Task {
 public:
  // constructor
  Task();
  virtual ~Task() = default;

  // ----- methods ----- //
  // returns an object which can compute the residual function.
  // the default implementation delegates to
  // Residual(mjModel*, mjData*, double*), for backwards compability, but
  // new implementations should return a custom ResidualFn object.
  virtual std::unique_ptr<ResidualFn> Residual() const {
    return std::make_unique<internal::ForwardingResidualFn>(this);
  }

  // should be overridden by subclasses to use internal ResidualFn
  virtual void Residual(const mjModel* model, const mjData* data,
                        double* residual) const = 0;

  // Must be called whenever parameters or weights change outside Transition or
  // Reset, so that calls to Residual use the new parameters.
  virtual void UpdateResidual() {}

  virtual void Transition(const mjModel* model, mjData* data) {}

  // get information from model
  virtual void Reset(const mjModel* model);

  virtual void ModifyScene(const mjModel* model, const mjData* data,
                           mjvScene* scene) const {}

  // compute cost terms
  virtual void CostTerms(double* terms, const double* residual,
                 bool weighted = true) const {
    return default_residual_.CostTerms(terms, residual, weighted);
  }

  // compute weighted cost
  virtual double CostValue(const double* residual) const {
    return default_residual_.CostValue(residual);
  }

  virtual std::string Name() const = 0;
  virtual std::string XmlPath() const = 0;

  // mode
  int mode;

  // GUI toggles
  int reset = 0;
  int visualize = 0;

  // cost parameters
  int num_residual;
  int num_term;
  int num_trace;
  std::vector<int> dim_norm_residual;
  std::vector<int> num_norm_parameter;
  std::vector<NormType> norm;
  std::vector<double> weight;
  std::vector<double> norm_parameter;
  double risk;

  // residual parameters
  std::vector<double> parameters;

 private:
  // initial residual parameters from model
  void SetFeatureParameters(const mjModel* model);
  internal::ForwardingResidualFn default_residual_;
};

// A version of Task which provides a Residual that can be run independently
// of the class, and where the parameters and weights used in the residual
// computations are guarded with a lock.
// TODO(nimrod): Migrate all tasks to this API, and deprecate the
// not-thread-safe Task.
class ThreadSafeTask : public Task {
 public:
  virtual ~ThreadSafeTask() override = default;

  // delegates to ResidualLocked, while holding a lock
  std::unique_ptr<ResidualFn> Residual() const final;

  // calls Residual on the pointer returned from InternalResidual(), while
  // holding a lock
  void Residual(const mjModel* model, const mjData* data,
                double* residual) const final;

  // Calls InternalResidual()->Update() with a lock.
  void UpdateResidual() final;

  // calls TransitionLocked and InternalResidual()->Update() while holding a
  // lock
  void Transition(const mjModel* model, mjData* data) final;

  // calls ResetLocked and InternalResidual()->Update() while holding a lock
  void Reset(const mjModel* model) final;

  // calls CostTerms on the pointer returned from InternalResidual(), while
  // holding a lock
  void CostTerms(double* terms, const double* residual,
                 bool weighted = true) const final;

  // calls CostValue on the pointer returned from InternalResidual(), while
  // holding a lock
  double CostValue(const double* residual) const final;

 protected:
  // returns a pointer to the ResidualFn instance that's used for physics
  // stepping and plotting, and is internal to the class
  virtual BaseResidualFn* InternalResidual() = 0;
  const BaseResidualFn* InternalResidual() const {
    return const_cast<ThreadSafeTask*>(this)->InternalResidual();
  }
  // returns an object which can compute the residual function. the function
  // can assume that a lock on mutex_ is held when it's called
  virtual std::unique_ptr<ResidualFn> ResidualLocked() const = 0;
  // implementation of Task::Transition() which can assume a lock is held
  virtual void TransitionLocked(const mjModel* model, mjData* data) {}
  // implementation of Task::Reset() which can assume a lock is held
  virtual void ResetLocked(const mjModel* model) {}
  // mutex which should be held on changes to InternalResidual.
  mutable std::mutex mutex_;
};

}  // namespace mjpc

#endif  // MJPC_TASK_H_
