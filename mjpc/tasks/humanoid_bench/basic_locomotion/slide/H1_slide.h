//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MUJOCO_MPC_H1_SLIDE_H
#define MUJOCO_MPC_H1_SLIDE_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_slide : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_slide *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_slide() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_slide_position : public H1_slide {
    public:
        std::string Name() const override {
            return "H1 Slide";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/slide/H1_slide_pos.xml");
        }
    };

    class H1_slide_hand : public H1_slide {
    public:
        std::string Name() const override {
            return "H1 Slide Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/slide/H1_slide_hand.xml");
        }
    };

    class H1_slide_gripper : public H1_slide {
    public:
        std::string Name() const override {
            return "H1 Slide Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/slide/H1_slide_gripper.xml");
        }
    };

    class H1_slide_simple_hand : public H1_slide {
    public:
        std::string Name() const override {
            return "H1 Slide Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/slide/H1_slide_simple_hand.xml");
        }
    };

    class H1_slide_strong : public H1_slide {
    public:
        std::string Name() const override {
            return "H1 Slide Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/slide/H1_slide_strong.xml");
        }
    };

    class H1_slide_touch : public H1_slide {
    public:
        std::string Name() const override {
            return "H1 Slide Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/slide/H1_slide_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MUJOCO_MPC_H1_SLIDE_H