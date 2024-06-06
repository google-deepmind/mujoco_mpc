#ifndef MUJOCO_MPC_H1_POWERLIFT_H
#define MUJOCO_MPC_H1_POWERLIFT_H
#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_powerlift : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_powerlift *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_powerlift() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_powerlift_position : public H1_powerlift {
    public:
        std::string Name() const override {
            return "H1 Powerlift Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/powerlift/H1_powerlift_pos.xml");
        }
    };

    class H1_powerlift_hand : public H1_powerlift {
    public:
        std::string Name() const override {
            return "H1 Powerlift Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/powerlift/H1_powerlift_hand.xml");
        }
    };

    class H1_powerlift_gripper : public H1_powerlift {
    public:
        std::string Name() const override {
            return "H1 Powerlift Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/powerlift/H1_powerlift_gripper.xml");
        }
    };

    class H1_powerlift_simple_hand : public H1_powerlift {
    public:
        std::string Name() const override {
            return "H1 Powerlift Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/powerlift/H1_powerlift_simple_hand.xml");
        }
    };

    class H1_powerlift_strong : public H1_powerlift {
    public:
        std::string Name() const override {
            return "H1 Powerlift Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/powerlift/H1_powerlift_strong.xml");
        }
    };

    class H1_powerlift_touch : public H1_powerlift {
    public:
        std::string Name() const override {
            return "H1 Powerlift Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/powerlift/H1_powerlift_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_POWERLIFT_H