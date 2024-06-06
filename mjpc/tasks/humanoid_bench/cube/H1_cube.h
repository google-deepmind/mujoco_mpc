#ifndef MUJOCO_MPC_H1_CUBE_H
#define MUJOCO_MPC_H1_CUBE_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_cube : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_cube *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_cube() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_cube_position : public H1_cube {
    public:
        std::string Name() const override {
            return "H1 Cube Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cube/H1_cube_pos.xml");
        }
    };

    class H1_cube_hand : public H1_cube {
    public:
        std::string Name() const override {
            return "H1 Cube Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cube/H1_cube_hand.xml");
        }
    };

    class H1_cube_gripper : public H1_cube {
    public:
        std::string Name() const override {
            return "H1 Cube Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cube/H1_cube_gripper.xml");
        }
    };

    class H1_cube_simple_hand : public H1_cube {
    public:
        std::string Name() const override {
            return "H1 Cube Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cube/H1_cube_simple_hand.xml");
        }
    };

    class H1_cube_strong : public H1_cube {
    public:
        std::string Name() const override {
            return "H1 Cube Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cube/H1_cube_strong.xml");
        }
    };

    class H1_cube_touch : public H1_cube {
    public:
        std::string Name() const override {
            return "H1 Cube Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cube/H1_cube_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_CUBE_H