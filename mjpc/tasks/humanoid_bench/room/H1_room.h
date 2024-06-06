#ifndef MUJOCO_MPC_H1_ROOM_H
#define MUJOCO_MPC_H1_ROOM_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_room : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_room *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_room() : residual_(this) {}

        void TransitionLocked(mjModel *model, mjData *data) override;

    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_room_position : public H1_room {
    public:
        std::string Name() const override {
            return "H1 Room Position";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/room/H1_room_pos.xml");
        }
    };

    class H1_room_hand : public H1_room {
    public:
        std::string Name() const override {
            return "H1 Room Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/room/H1_room_hand.xml");
        }
    };

    class H1_room_gripper : public H1_room {
    public:
        std::string Name() const override {
            return "H1 Room Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/room/H1_room_gripper.xml");
        }
    };

    class H1_room_simple_hand : public H1_room {
    public:
        std::string Name() const override {
            return "H1 Room Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/room/H1_room_simple_hand.xml");
        }
    };

    class H1_room_strong : public H1_room {
    public:
        std::string Name() const override {
            return "H1 Room Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/room/H1_room_strong.xml");
        }
    };

    class H1_room_touch : public H1_room {
    public:
        std::string Name() const override {
            return "H1 Room Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/room/H1_room_touch.xml");
        }
    };
}  // namespace mjpc

#endif //MUJOCO_MPC_H1_ROOM_H