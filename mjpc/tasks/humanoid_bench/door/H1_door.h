#ifndef MUJOCO_MPC_H1_DOOR_H
#define MUJOCO_MPC_H1_DOOR_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_door : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_door *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_door() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_door_position : public H1_door {
    public:
        std::string Name() const override {
            return "H1 Door Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/door/H1_door_pos.xml");
        }
    };

    class H1_door_hand : public H1_door {
    public:
        std::string Name() const override {
            return "H1 Door Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/door/H1_door_hand.xml");
        }
    };

    class H1_door_gripper : public H1_door {
    public:
        std::string Name() const override {
            return "H1 Door Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/door/H1_door_gripper.xml");
        }
    };

    class H1_door_simple_hand : public H1_door {
    public:
        std::string Name() const override {
            return "H1 Door Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/door/H1_door_simple_hand.xml");
        }
    };

    class H1_door_strong : public H1_door {
    public:
        std::string Name() const override {
            return "H1 Door Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/door/H1_door_strong.xml");
        }
    };

    class H1_door_touch : public H1_door {
    public:
        std::string Name() const override {
            return "H1 Door Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/door/H1_door_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_DOOR_H