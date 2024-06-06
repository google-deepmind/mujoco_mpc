#ifndef MJPC_TASKS_H1_REACH_H_
#define MJPC_TASKS_H1_REACH_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_reach : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_reach *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_reach() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_reach_position : public H1_reach {
    public:
        std::string Name() const override {
            return "H1 Reach Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/reach/H1_reach_pos.xml");
        }
    };

    class H1_reach_hand : public H1_reach {
    public:
        std::string Name() const override {
            return "H1 Reach Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/reach/H1_reach_hand.xml");
        }
    };

    class H1_reach_gripper : public H1_reach {
    public:
        std::string Name() const override {
            return "H1 Reach Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/reach/H1_reach_gripper.xml");
        }
    };

    class H1_reach_simple_hand : public H1_reach {
    public:
        std::string Name() const override {
            return "H1 Reach Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/reach/H1_reach_simple_hand.xml");
        }
    };

    class H1_reach_strong : public H1_reach {
    public:
        std::string Name() const override {
            return "H1 Reach Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/reach/H1_reach_strong.xml");
        }
    };

    class H1_reach_touch : public H1_reach {
    public:
        std::string Name() const override {
            return "H1 Reach Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/reach/H1_reach_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_REACH_H_