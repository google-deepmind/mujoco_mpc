#ifndef MJPC_TASKS_H1_BALANCE_SIMPLE_H_
#define MJPC_TASKS_H1_BALANCE_SIMPLE_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class Balance_Simple : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const Balance_Simple *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        Balance_Simple() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class Balance_Simple_position : public Balance_Simple {
    public:
        std::string Name() const override {
            return "Balance Simple Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/balance/H1_balance_pos.xml");
        }
    };

    class Balance_Simple_hand : public Balance_Simple {
    public:
        std::string Name() const override {
            return "Balance Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/balance/H1_balance_hand.xml");
        }
    };

    class Balance_Simple_gripper : public Balance_Simple {
    public:
        std::string Name() const override {
            return "Balance Simple Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/balance/H1_balance_gripper.xml");
        }
    };

    class Balance_Simple_simple_hand : public Balance_Simple {
    public:
        std::string Name() const override {
            return "Balance Simple Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/balance/H1_balance_simple_hand.xml");
        }
    };

    class Balance_Simple_strong : public Balance_Simple {
    public:
        std::string Name() const override {
            return "Balance Simple Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/balance/H1_balance_strong.xml");
        }
    };

    class Balance_Simple_touch : public Balance_Simple {
    public:
        std::string Name() const override {
            return "Balance Simple Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/balance/H1_balance_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_BALANCE_SIMPLE_H_