#ifndef MUJOCO_MPC_H1_BOOKSHELF_H
#define MUJOCO_MPC_H1_BOOKSHELF_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_bookshelf : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_bookshelf *task) : mjpc::BaseResidualFn(task),
                                                            task_(const_cast<H1_bookshelf *>(task)) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;

        private:
            H1_bookshelf *task_;
        };

        H1_bookshelf() : residual_(this), task_index_(0) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
        int task_index_;
    };

    class H1_bookshelf_position : public H1_bookshelf {
    public:
        std::string Name() const override {
            return "H1 Bookshelf Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/bookshelf/H1_bookshelf_pos.xml");
        }
    };

    class H1_bookshelf_hand : public H1_bookshelf {
    public:
        std::string Name() const override {
            return "H1 Bookshelf Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/bookshelf/H1_bookshelf_hand.xml");
        }
    };

    class H1_bookshelf_gripper : public H1_bookshelf {
    public:
        std::string Name() const override {
            return "H1 Bookshelf Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/bookshelf/H1_bookshelf_gripper.xml");
        }
    };

    class H1_bookshelf_simple_hand : public H1_bookshelf {
    public:
        std::string Name() const override {
            return "H1 Bookshelf Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/bookshelf/H1_bookshelf_simple_hand.xml");
        }
    };

    class H1_bookshelf_strong : public H1_bookshelf {
    public:
        std::string Name() const override {
            return "H1 Bookshelf Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/bookshelf/H1_bookshelf_strong.xml");
        }
    };

    class H1_bookshelf_touch : public H1_bookshelf {
    public:
        std::string Name() const override {
            return "H1 Bookshelf Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/bookshelf/H1_bookshelf_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_BOOKSHELF_H