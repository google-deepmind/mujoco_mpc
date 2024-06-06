//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MJPC_TASKS_H1_WALK_H_
#define MJPC_TASKS_H1_WALK_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"


namespace mjpc {
    class H1_walk : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_walk *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_walk() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_walk_position : public H1_walk {
    public:
        std::string Name() const override {
            return "H1 Walk";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/walk/H1_walk_position.xml");
        }
    };

    class H1_walk_hand : public H1_walk {
    public:
        std::string Name() const override {
            return "H1 Walk Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/walk/H1_walk_hand.xml");
        }

    };

    class H1_walk_gripper : public H1_walk {
    public:
        std::string Name() const override {
            return "H1 Walk Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/walk/H1_walk_gripper.xml");
        }
    };

    class H1_walk_simple_hand : public H1_walk {
    public:
        std::string Name() const override {
            return "H1 Walk Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/walk/H1_walk_simple_hand.xml");
        }
    };

    class H1_walk_strong : public H1_walk {
    public:
        std::string Name() const override {
            return "H1 Walk Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/walk/H1_walk_strong.xml");
        }
    };

    class H1_walk_touch : public H1_walk {
    public:
        std::string Name() const override {
            return "H1 Walk Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/walk/H1_walk_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_WALK_H_
