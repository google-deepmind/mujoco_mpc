//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MJPC_TASKS_H1_STAIRS_H_
#define MJPC_TASKS_H1_STAIRS_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_stairs : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_stairs *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_stairs() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_stairs_position : public H1_stairs {
    public:
        std::string Name() const override {
            return "H1 Stairs";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stairs/H1_stairs_pos.xml");
        }
    };

    class H1_stairs_hand : public H1_stairs {
    public:
        std::string Name() const override {
            return "H1 Stairs Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stairs/H1_stairs_hand.xml");
        }
    };

    class H1_stairs_gripper : public H1_stairs {
    public:
        std::string Name() const override {
            return "H1 Stairs Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stairs/H1_stairs_gripper.xml");
        }
    };

    class H1_stairs_simple_hand : public H1_stairs {
    public:
        std::string Name() const override {
            return "H1 Stairs Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stairs/H1_stairs_simple_hand.xml");
        }
    };

    class H1_stairs_strong : public H1_stairs {
    public:
        std::string Name() const override {
            return "H1 Stairs Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stairs/H1_stairs_strong.xml");
        }
    };

    class H1_stairs_touch : public H1_stairs {
    public:
        std::string Name() const override {
            return "H1 Stairs Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stairs/H1_stairs_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_STAIRS_H_