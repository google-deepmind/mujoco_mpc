//
// Created by Moritz Meser on 20.05.24.
//

#ifndef MUJOCO_MPC_H1_SIT_H
#define MUJOCO_MPC_H1_SIT_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_sit : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_sit *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_sit() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_sit_position : public H1_sit {
    public:
        std::string Name() const override {
            return "H1 Sit";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/sit/H1_sit_pos.xml");
        }
    };

    class H1_sit_hand : public H1_sit {
    public:
        std::string Name() const override {
            return "H1 Sit Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/sit/H1_sit_hand.xml");
        }
    };

    class H1_sit_gripper : public H1_sit {
    public:
        std::string Name() const override {
            return "H1 Sit Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/sit/H1_sit_gripper.xml");
        }
    };

    class H1_sit_simple_hand : public H1_sit {
    public:
        std::string Name() const override {
            return "H1 Sit Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/sit/H1_sit_simple_hand.xml");
        }
    };

    class H1_sit_strong : public H1_sit {
    public:
        std::string Name() const override {
            return "H1 Sit Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/sit/H1_sit_strong.xml");
        }
    };

    class H1_sit_touch : public H1_sit {
    public:
        std::string Name() const override {
            return "H1 Sit Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/sit/H1_sit_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_SIT_H