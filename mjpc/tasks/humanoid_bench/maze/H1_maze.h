//
// Created by Moritz Meser on 20.05.24.
//

#ifndef MUJOCO_MPC_H1_MAZE_H
#define MUJOCO_MPC_H1_MAZE_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_maze : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_maze *task) : mjpc::BaseResidualFn(task),
                                                       task_(const_cast<H1_maze *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            const H1_maze *task_;
        };

        H1_maze() : residual_(this), curr_goal_idx_(0) {}

// -------- Transition for humanoid_bench Maze task -------- //
// ------------------------------------------------------------ //
        void TransitionLocked(mjModel *model, mjData *data) override;

        // call base-class Reset, save task-related ids
        void ResetLocked(const mjModel *model) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        int curr_goal_idx_;
    };

    class H1_maze_position : public H1_maze {
    public:
        std::string Name() const override {
            return "H1 Maze Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/maze/H1_maze_pos.xml");
        }
    };

    class H1_maze_hand : public H1_maze {
    public:
        std::string Name() const override {
            return "H1 Maze Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/maze/H1_maze_hand.xml");
        }
    };

    class H1_maze_gripper : public H1_maze {
    public:
        std::string Name() const override {
            return "H1 Maze Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/maze/H1_maze_gripper.xml");
        }
    };

    class H1_maze_simple_hand : public H1_maze {
    public:
        std::string Name() const override {
            return "H1 Maze Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/maze/H1_maze_simple_hand.xml");
        }
    };

    class H1_maze_strong : public H1_maze {
    public:
        std::string Name() const override {
            return "H1 Maze Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/maze/H1_maze_strong.xml");
        }
    };

    class H1_maze_touch : public H1_maze {
    public:
        std::string Name() const override {
            return "H1 Maze Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/maze/H1_maze_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_MAZE_H