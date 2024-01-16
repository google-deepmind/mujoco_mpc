#ifndef MJPC_TASKS_ALLEGROCUBE_ALLEGROCUBE_H_
#define MJPC_TASKS_ALLEGROCUBE_ALLEGROCUBE_H_

#include <string>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"

namespace mjpc {
class AllegroCube : public Task {
 public:
  std::string Name() const override;
  std::string XmlPath() const override;

  class ResidualFn : public BaseResidualFn {
   public:
    explicit ResidualFn(const AllegroCube* task) : BaseResidualFn(task) {}
    void Residual(const mjModel* model, const mjData* data,
                  double* residual) const override;
  };
  AllegroCube() : residual_(this) {}

  // Reset the cube into the hand if it's on the floor
  void TransitionLocked(mjModel* model, mjData* data) override;

 protected:
  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(this);
  }
  ResidualFn* InternalResidual() override { return &residual_; }

 private:
  ResidualFn residual_;
};
}  // namespace mjpc

#endif  // MJPC_TASKS_ALLEGROCUBE_ALLEGROCUBE_H_
