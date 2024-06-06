//
// Created by Moritz Meser on 21.05.24.
//

#ifndef MUJOCO_MPC_H1_KITCHEN_H
#define MUJOCO_MPC_H1_KITCHEN_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_kitchen : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        static double CalculateDistance(const std::string &task, const mjData *data, int robot_dof) ;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_kitchen *task) : mjpc::BaseResidualFn(task),
                                                          task_(const_cast<H1_kitchen *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            H1_kitchen *task_;
        };

        H1_kitchen() : residual_(this),
                       REMOVE_TASKS_WHEN_COMPLETE(true),
//                       TERMINATE_ON_TASK_COMPLETE(true),
                       ENFORCE_TASK_ORDER(true),
                       tasks_to_complete_({"microwave", "kettle", "bottom burner", "light switch"}) {}

// -------- Transition for humanoid_bench kitchen task -------- //
// ------------------------------------------------------------ //
        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        bool REMOVE_TASKS_WHEN_COMPLETE;
//        bool TERMINATE_ON_TASK_COMPLETE;
        bool ENFORCE_TASK_ORDER;
        std::vector<std::string> tasks_to_complete_;
    };
    class H1_kitchen_position : public H1_kitchen {
    public:
        std::string Name() const override {
            return "H1 Kitchen Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/kitchen/H1_kitchen_pos.xml");
        }
    };

    class H1_kitchen_hand : public H1_kitchen {
    public:
        std::string Name() const override {
            return "H1 Kitchen Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/kitchen/H1_kitchen_hand.xml");
        }
    };

    class H1_kitchen_gripper : public H1_kitchen {
    public:
        std::string Name() const override {
            return "H1 Kitchen Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/kitchen/H1_kitchen_gripper.xml");
        }
    };

    class H1_kitchen_simple_hand : public H1_kitchen {
    public:
        std::string Name() const override {
            return "H1 Kitchen Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/kitchen/H1_kitchen_simple_hand.xml");
        }
    };

    class H1_kitchen_strong : public H1_kitchen {
    public:
        std::string Name() const override {
            return "H1 Kitchen Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/kitchen/H1_kitchen_strong.xml");
        }
    };

    class H1_kitchen_touch : public H1_kitchen {
    public:
        std::string Name() const override {
            return "H1 Kitchen Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/kitchen/H1_kitchen_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_KITCHEN_H