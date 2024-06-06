#ifndef MUJOCO_MPC_H1_SPOON_H
#define MUJOCO_MPC_H1_SPOON_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_spoon : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_spoon *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_spoon() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_spoon_position : public H1_spoon {
    public:
        std::string Name() const override {
            return "H1 Spoon Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/spoon/H1_spoon_pos.xml");
        }
    };

    class H1_spoon_hand : public H1_spoon {
    public:
        std::string Name() const override {
            return "H1 Spoon Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/spoon/H1_spoon_hand.xml");
        }
    };

    class H1_spoon_gripper : public H1_spoon {
    public:
        std::string Name() const override {
            return "H1 Spoon Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/spoon/H1_spoon_gripper.xml");
        }
    };

    class H1_spoon_simple_hand : public H1_spoon {
    public:
        std::string Name() const override {
            return "H1 Spoon Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/spoon/H1_spoon_simple_hand.xml");
        }
    };

    class H1_spoon_strong : public H1_spoon {
    public:
        std::string Name() const override {
            return "H1 Spoon Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/spoon/H1_spoon_strong.xml");
        }
    };

    class H1_spoon_touch : public H1_spoon {
    public:
        std::string Name() const override {
            return "H1 Spoon Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/spoon/H1_spoon_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_SPOON_H