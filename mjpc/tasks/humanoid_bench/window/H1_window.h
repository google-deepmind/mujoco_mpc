#ifndef MUJOCO_MPC_H1_WINDOW_H
#define MUJOCO_MPC_H1_WINDOW_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_window : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_window *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_window() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_window_position : public H1_window {
    public:
        std::string Name() const override {
            return "H1 Window Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/window/H1_window_pos.xml");
        }
    };

    class H1_window_hand : public H1_window {
    public:
        std::string Name() const override {
            return "H1 Window Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/window/H1_window_hand.xml");
        }
    };

    class H1_window_gripper : public H1_window {
    public:
        std::string Name() const override {
            return "H1 Window Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/window/H1_window_gripper.xml");
        }
    };

    class H1_window_simple_hand : public H1_window {
    public:
        std::string Name() const override {
            return "H1 Window Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/window/H1_window_simple_hand.xml");
        }
    };

    class H1_window_strong : public H1_window {
    public:
        std::string Name() const override {
            return "H1 Window Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/window/H1_window_strong.xml");
        }
    };

    class H1_window_touch : public H1_window {
    public:
        std::string Name() const override {
            return "H1 Window Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/window/H1_window_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_WINDOW_H