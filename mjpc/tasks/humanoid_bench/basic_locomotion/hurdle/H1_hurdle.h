//
// Created by Moritz Meser on 16.05.24.
//

#ifndef MUJOCO_MPC_H1_HURDLE_H
#define MUJOCO_MPC_H1_HURDLE_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_hurdle : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_hurdle *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_hurdle() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_hurdle_position : public H1_hurdle {
    public:
        std::string Name() const override {
            return "H1 Hurdle";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/hurdle/H1_hurdle_pos.xml");
        }
    };

    class H1_hurdle_hand : public H1_hurdle {
    public:
        std::string Name() const override {
            return "H1 Hurdle Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/hurdle/H1_hurdle_hand.xml");
        }
    };

    class H1_hurdle_gripper : public H1_hurdle {
    public:
        std::string Name() const override {
            return "H1 Hurdle Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/hurdle/H1_hurdle_gripper.xml");
        }
    };

    class H1_hurdle_simple_hand : public H1_hurdle {
    public:
        std::string Name() const override {
            return "H1 Hurdle Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/hurdle/H1_hurdle_simple_hand.xml");
        }
    };

    class H1_hurdle_strong : public H1_hurdle {
    public:
        std::string Name() const override {
            return "H1 Hurdle Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/hurdle/H1_hurdle_strong.xml");
        }
    };

    class H1_hurdle_touch : public H1_hurdle {
    public:
        std::string Name() const override {
            return "H1 Hurdle Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/hurdle/H1_hurdle_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MUJOCO_MPC_H1_HURDLE_H