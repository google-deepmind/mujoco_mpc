#ifndef MUJOCO_MPC_H1_INSERT_H
#define MUJOCO_MPC_H1_INSERT_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_insert : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_insert *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_insert() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_insert_position : public H1_insert {
    public:
        std::string Name() const override {
            return "H1 Insert Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/insert/H1_insert_pos.xml");
        }
    };

    class H1_insert_hand : public H1_insert {
    public:
        std::string Name() const override {
            return "H1 Insert Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/insert/H1_insert_hand.xml");
        }
    };

    class H1_insert_gripper : public H1_insert {
    public:
        std::string Name() const override {
            return "H1 Insert Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/insert/H1_insert_gripper.xml");
        }
    };

    class H1_insert_simple_hand : public H1_insert {
    public:
        std::string Name() const override {
            return "H1 Insert Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/insert/H1_insert_simple_hand.xml");
        }
    };

    class H1_insert_strong : public H1_insert {
    public:
        std::string Name() const override {
            return "H1 Insert Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/insert/H1_insert_strong.xml");
        }
    };

    class H1_insert_touch : public H1_insert {
    public:
        std::string Name() const override {
            return "H1 Insert Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/insert/H1_insert_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_INSERT_H