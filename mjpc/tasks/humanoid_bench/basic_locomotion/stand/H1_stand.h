//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MJPC_TASKS_H1_STAND_H_
#define MJPC_TASKS_H1_STAND_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_stand : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_stand *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_stand() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_stand_position : public H1_stand {
    public:
        std::string Name() const override {
            return "H1 Stand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stand/H1_stand_pos.xml");
        }
    };

    class H1_stand_hand : public H1_stand {
    public:
        std::string Name() const override {
            return "H1 Stand Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stand/H1_stand_hand.xml");
        }
    };

    class H1_stand_gripper : public H1_stand {
    public:
        std::string Name() const override {
            return "H1 Stand Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stand/H1_stand_gripper.xml");
        }
    };

    class H1_stand_simple_hand : public H1_stand {
    public:
        std::string Name() const override {
            return "H1 Stand Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stand/H1_stand_simple_hand.xml");
        }
    };

    class H1_stand_strong : public H1_stand {
    public:
        std::string Name() const override {
            return "H1 Stand Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stand/H1_stand_strong.xml");
        }
    };

    class H1_stand_touch : public H1_stand {
    public:
        std::string Name() const override {
            return "H1 Stand Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/stand/H1_stand_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_STAND_H_