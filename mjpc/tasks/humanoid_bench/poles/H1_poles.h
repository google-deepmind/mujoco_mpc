#ifndef MUJOCO_MPC_H1_POLES_H
#define MUJOCO_MPC_H1_POLES_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_poles : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_poles *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_poles() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_poles_position : public H1_poles {
    public:
        std::string Name() const override {
            return "H1 Poles Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/poles/H1_poles_pos.xml");
        }
    };

    class H1_poles_hand : public H1_poles {
    public:
        std::string Name() const override {
            return "H1 Poles Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/poles/H1_poles_hand.xml");
        }
    };

    class H1_poles_gripper : public H1_poles {
    public:
        std::string Name() const override {
            return "H1 Poles Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/poles/H1_poles_gripper.xml");
        }
    };

    class H1_poles_simple_hand : public H1_poles {
    public:
        std::string Name() const override {
            return "H1 Poles Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/poles/H1_poles_simple_hand.xml");
        }
    };

    class H1_poles_strong : public H1_poles {
    public:
        std::string Name() const override {
            return "H1 Poles Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/poles/H1_poles_strong.xml");
        }
    };

    class H1_poles_touch : public H1_poles {
    public:
        std::string Name() const override {
            return "H1 Poles Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/poles/H1_poles_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_POLES_H