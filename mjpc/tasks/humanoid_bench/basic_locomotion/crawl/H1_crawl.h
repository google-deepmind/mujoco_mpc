//
// Created by Moritz Meser on 15.05.24.
//

#ifndef MUJOCO_MPC_H1_CRAWL_H
#define MUJOCO_MPC_H1_CRAWL_H

#include <string>
#include "mujoco/mujoco.h"
#include "mjpc/task.h"
#include "mjpc/utilities.h"

namespace mjpc {
    class H1_crawl : public Task {
    public:
        std::string Name() const override = 0;

        std::string XmlPath() const override = 0;

        class ResidualFn : public mjpc::BaseResidualFn {
        public:
            explicit ResidualFn(const H1_crawl *task) : mjpc::BaseResidualFn(task) {}

            void Residual(const mjModel *model, const mjData *data,
                          double *residual) const override;
        };

        H1_crawl() : residual_(this) {}


    protected:
        std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
            return std::make_unique<ResidualFn>(this);
        }

        ResidualFn *InternalResidual() override { return &residual_; }

    private:
        ResidualFn residual_;
    };

    class H1_crawl_position : public H1_crawl {
    public:
        std::string Name() const override {
            return "H1 Crawl";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/crawl/H1_crawl_pos.xml");
        }
    };

    class H1_crawl_hand : public H1_crawl {
    public:
        std::string Name() const override {
            return "H1 Crawl Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/crawl/H1_crawl_hand.xml");
        }
    };

    class H1_crawl_gripper : public H1_crawl {
    public:
        std::string Name() const override {
            return "H1 Crawl Gripper";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/crawl/H1_crawl_gripper.xml");
        }
    };

    class H1_crawl_simple_hand : public H1_crawl {
    public:
        std::string Name() const override {
            return "H1 Crawl Simple Hand";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/crawl/H1_crawl_simple_hand.xml");
        }
    };

    class H1_crawl_strong : public H1_crawl {
    public:
        std::string Name() const override {
            return "H1 Crawl Strong";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/crawl/H1_crawl_strong.xml");
        }
    };

    class H1_crawl_touch : public H1_crawl {
    public:
        std::string Name() const override {
            return "H1 Crawl Touch";
        }

        std::string XmlPath() const override {
            return GetModelPath("humanoid_bench/basic_locomotion/crawl/H1_crawl_touch.xml");
        }
    };
}  // namespace mjpc

#endif  // MUJOCO_MPC_H1_CRAWL_H