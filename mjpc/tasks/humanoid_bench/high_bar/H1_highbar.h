#ifndef MUJOCO_MPC_H1_HIGHBAR_H
#define MUJOCO_MPC_H1_HIGHBAR_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_highbar : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_highbar *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_highbar() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_highbar_position : public H1_highbar {
    public:
        std::string Name() const override {
            return "H1 Highbar Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/high_bar/H1_high_bar_pos.xml");
        }
    };

    class H1_highbar_hand : public H1_highbar {
    public:
        std::string Name() const override {
            return "H1 Highbar Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/high_bar/H1_high_bar_hand.xml");
        }
    };

    class H1_highbar_gripper : public H1_highbar {
    public:
        std::string Name() const override {
            return "H1 Highbar Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/high_bar/H1_high_bar_gripper.xml");
        }
    };

    class H1_highbar_simple_hand : public H1_highbar {
    public:
        std::string Name() const override {
            return "H1 Highbar Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/high_bar/H1_high_bar_simple_hand.xml");
        }
    };

    class H1_highbar_strong : public H1_highbar {
    public:
        std::string Name() const override {
            return "H1 Highbar Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/high_bar/H1_high_bar_strong.xml");
        }
    };

    class H1_highbar_touch : public H1_highbar {
    public:
        std::string Name() const override {
            return "H1 Highbar Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/high_bar/H1_high_bar_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_HIGHBAR_H