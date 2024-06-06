//
// Created by Moritz Meser on 21.05.24.
//

#ifndef MUJOCO_MPC_H1_CABINET_H
#define MUJOCO_MPC_H1_CABINET_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_cabinet : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_cabinet *task) : mjpc::BaseResidualFn(task), task_(
                    const_cast<H1_cabinet *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            H1_cabinet *task_;
        };


        H1_cabinet() : residual_(this), current_subtask_(1) {}

// -------- Transition for humanoid_bench cabinet task -------- //
// ------------------------------------------------------------ //
        void TransitionLocked(mjModel *model, mjData *data) override;

        // call base-class Reset, save task-related ids
        void ResetLocked(const mjModel* model) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        mutable int current_subtask_;
    };

    class H1_cabinet_position : public H1_cabinet {
    public:
        std::string Name() const override {
            return "H1 Cabinet Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cabinet/H1_cabinet_pos.xml");
        }
    };

    class H1_cabinet_hand : public H1_cabinet {
    public:
        std::string Name() const override {
            return "H1 Cabinet Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cabinet/H1_cabinet_hand.xml");
        }
    };

    class H1_cabinet_gripper : public H1_cabinet {
    public:
        std::string Name() const override {
            return "H1 Cabinet Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cabinet/H1_cabinet_gripper.xml");
        }
    };

    class H1_cabinet_simple_hand : public H1_cabinet {
    public:
        std::string Name() const override {
            return "H1 Cabinet Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cabinet/H1_cabinet_simple_hand.xml");
        }
    };

    class H1_cabinet_strong : public H1_cabinet {
    public:
        std::string Name() const override {
            return "H1 Cabinet Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cabinet/H1_cabinet_strong.xml");
        }
    };

    class H1_cabinet_touch : public H1_cabinet {
    public:
        std::string Name() const override {
            return "H1 Cabinet Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/cabinet/H1_cabinet_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_CABINET_H