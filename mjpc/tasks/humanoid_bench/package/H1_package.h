#ifndef MUJOCO_MPC_H1_PACKAGE_H
#define MUJOCO_MPC_H1_PACKAGE_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_package : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_package *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_package() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_package_position : public H1_package {
    public:
        std::string Name() const override {
            return "H1 Package Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/package/H1_package_pos.xml");
        }
    };

    class H1_package_hand : public H1_package {
    public:
        std::string Name() const override {
            return "H1 Package Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/package/H1_package_hand.xml");
        }
    };

    class H1_package_gripper : public H1_package {
    public:
        std::string Name() const override {
            return "H1 Package Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/package/H1_package_gripper.xml");
        }
    };

    class H1_package_simple_hand : public H1_package {
    public:
        std::string Name() const override {
            return "H1 Package Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/package/H1_package_simple_hand.xml");
        }
    };

    class H1_package_strong : public H1_package {
    public:
        std::string Name() const override {
            return "H1 Package Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/package/H1_package_strong.xml");
        }
    };

    class H1_package_touch : public H1_package {
    public:
        std::string Name() const override {
            return "H1 Package Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/package/H1_package_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_PACKAGE_H