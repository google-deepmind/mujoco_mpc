//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MJPC_TASKS_H1_RUN_H_
#define MJPC_TASKS_H1_RUN_H_

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_run : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_run *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_run() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_run_position : public H1_run {
    public:
        std::string Name() const override {
            return "H1 Run";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/run/H1_run_pos.xml");
        }
    };

    class H1_run_hand : public H1_run {
    public:
        std::string Name() const override {
            return "H1 Run Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/run/H1_run_hand.xml");
        }
    };

    class H1_run_gripper : public H1_run {
    public:
        std::string Name() const override {
            return "H1 Run Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/run/H1_run_gripper.xml");
        }
    };

    class H1_run_simple_hand : public H1_run {
    public:
        std::string Name() const override {
            return "H1 Run Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/run/H1_run_simple_hand.xml");
        }
    };

    class H1_run_strong : public H1_run {
    public:
        std::string Name() const override {
            return "H1 Run Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/run/H1_run_strong.xml");
        }
    };

    class H1_run_touch : public H1_run {
    public:
        std::string Name() const override {
            return "H1 Run Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/run/H1_run_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MJPC_TASKS_H1_RUN_H_