#ifndef MUJOCO_MPC_H1_BASKETBALL_H
#define MUJOCO_MPC_H1_BASKETBALL_H
#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/tasks/humanoid_bench/utility/dm_control_utils_rewards.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_basketball : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_basketball *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_basketball() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_basketball_position : public H1_basketball {
    public:
        std::string Name() const override {
            return "H1 Basketball Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basketball/H1_basketball_pos.xml");
        }
    };

    class H1_basketball_hand : public H1_basketball {
    public:
        std::string Name() const override {
            return "H1 Basketball Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basketball/H1_basketball_hand.xml");
        }
    };

    class H1_basketball_gripper : public H1_basketball {
    public:
        std::string Name() const override {
            return "H1 Basketball Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basketball/H1_basketball_gripper.xml");
        }
    };

    class H1_basketball_simple_hand : public H1_basketball {
    public:
        std::string Name() const override {
            return "H1 Basketball Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basketball/H1_basketball_simple_hand.xml");
        }
    };

    class H1_basketball_strong : public H1_basketball {
    public:
        std::string Name() const override {
            return "H1 Basketball Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basketball/H1_basketball_strong.xml");
        }
    };

    class H1_basketball_touch : public H1_basketball {
    public:
        std::string Name() const override {
            return "H1 Basketball Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basketball/H1_basketball_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_BASKETBALL_H