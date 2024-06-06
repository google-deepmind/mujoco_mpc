#ifndef MUJOCO_MPC_H1_PUSH_H
#define MUJOCO_MPC_H1_PUSH_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_push : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_push *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_push() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_push_position : public H1_push {
    public:
        std::string Name() const override {
            return "H1 Push Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/push/H1_push_pos.xml");
        }
    };

    class H1_push_hand : public H1_push {
    public:
        std::string Name() const override {
            return "H1 Push Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/push/H1_push_hand.xml");
        }
    };

    class H1_push_gripper : public H1_push {
    public:
        std::string Name() const override {
            return "H1 Push Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/push/H1_push_gripper.xml");
        }
    };

    class H1_push_simple_hand : public H1_push {
    public:
        std::string Name() const override {
            return "H1 Push Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/push/H1_push_simple_hand.xml");
        }
    };

    class H1_push_strong : public H1_push {
    public:
        std::string Name() const override {
            return "H1 Push Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/push/H1_push_strong.xml");
        }
    };

    class H1_push_touch : public H1_push {
    public:
        std::string Name() const override {
            return "H1 Push Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/push/H1_push_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_PUSH_H