// Copyright 2022 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/agent.h"

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/test/load.h"
#include "mjpc/test/testdata/particle_residual.h"
#include "mjpc/threadpool.h"

namespace mjpc {
namespace {
mjModel* model = nullptr;
Agent* agent = nullptr;
}  // namespace

class AgentTest : public ::testing::Test {
 protected:
  void SetUp() override {
    agent = new Agent;
    std::vector<std::shared_ptr<Task>> tasks;
    tasks.push_back(std::make_unique<ParticleTestTask>());
    agent->SetTaskList(std::move(tasks));
  }
  void TearDown() override {
    delete agent;
    agent = nullptr;
    mjcb_sensor = nullptr;
  }

  static void SensorCallback(const mjModel* model, mjData* data, int stage) {
    if (stage == mjSTAGE_ACC) {
      agent->ActiveTask()->Residual(model, data, data->sensordata);
    }
  }

  // test agent initialization
  void TestInitialization() {
    // load model
    model = LoadTestModel("particle_task.xml");

    // create data
    mjData* data = mj_makeData(model);

    // ----- initialize agent ----- //
    agent->Initialize(model);

    // sensor callback
    mjcb_sensor = &SensorCallback;

    // test
    EXPECT_EQ(agent->integrator_, 0);
    EXPECT_NEAR(agent->timestep_, 0.1, 1.0e-5);
    EXPECT_EQ(agent->planner_, 0);
    EXPECT_EQ(agent->state_, 0);
    EXPECT_NEAR(agent->horizon_, 1.0, 1.0e-5);
    EXPECT_EQ(agent->steps_, 11);
    EXPECT_FALSE(agent->plan_enabled);
    EXPECT_TRUE(agent->action_enabled);
    EXPECT_FALSE(agent->visualize_enabled);
    EXPECT_TRUE(agent->allocate_enabled);
    EXPECT_TRUE(agent->plot_enabled);

    // allocate
    agent->Allocate();

    // test
    EXPECT_FALSE(agent->allocate_enabled);

    // delete data
    mj_deleteData(data);

    // delete model
    mj_deleteModel(model);
  }

  void TestPlan() {
    // load model
    model = LoadTestModel("particle_task.xml");

    // create data
    mjData* data = mj_makeData(model);

    // sensor callback
    mjcb_sensor = &SensorCallback;

    // ----- initialize agent ----- //
    agent->Initialize(model);
    agent->Allocate();
    agent->Reset();

    // pool
    ThreadPool plan_pool(1);

    // ----- settings ----- //
    std::atomic<bool> exitrequest(false);
    std::atomic<int> uiloadrequest(0);
    agent->plan_enabled = true;

    // ----- plan w/ random search ----- //
    plan_pool.Schedule([&exitrequest, &uiloadrequest]() {
      agent->Plan(exitrequest, uiloadrequest);
    });

    // wait
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (agent->count_ > 1000) {
        exitrequest.store(true);
        break;
      }
    }
    plan_pool.WaitCount(1);
    plan_pool.ResetCount();

    // test final state
    EXPECT_NEAR(agent->ActivePlanner()
                    .BestTrajectory()
                    ->states[(agent->steps_ - 1) *
                             (agent->model_->nq + agent->model_->nv)],
                agent->states_[agent->state_]->mocap()[0], 1.0e-1);
    EXPECT_NEAR(
        agent->ActivePlanner().BestTrajectory()->states
            [(agent->steps_ - 1) * (agent->model_->nq + agent->model_->nv) + 1],
        agent->states_[agent->state_]->mocap()[1], 1.0e-1);
    EXPECT_NEAR(
        agent->ActivePlanner().BestTrajectory()->states
            [(agent->steps_ - 1) * (agent->model_->nq + agent->model_->nv) + 2],
        0.0, 1.0e-1);
    EXPECT_NEAR(
        agent->ActivePlanner().BestTrajectory()->states
            [(agent->steps_ - 1) * (agent->model_->nq + agent->model_->nv) + 3],
        0.0, 1.0e-1);

    // ----- switch to iLQG planner ----- //
    agent->planner_ = 2;
    agent->Allocate();
    agent->Reset();
    exitrequest.store(false);

    // ----- plan w/ iLQG planner ----- //
    agent->plan_enabled = true;

    plan_pool.Schedule([&exitrequest, &uiloadrequest]() {
      agent->Plan(exitrequest, uiloadrequest);
    });

    // wait
    while (true) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      if (agent->count_ > 1000) {
        exitrequest.store(true);
        break;
      }
    }
    plan_pool.WaitCount(1);
    plan_pool.ResetCount();

    // test final state
    EXPECT_NEAR(agent->ActivePlanner()
                    .BestTrajectory()
                    ->states[(agent->steps_ - 1) *
                             (agent->model_->nq + agent->model_->nv)],
                agent->states_[agent->state_]->mocap()[0], 1.0e-1);
    EXPECT_NEAR(
        agent->ActivePlanner().BestTrajectory()->states
            [(agent->steps_ - 1) * (agent->model_->nq + agent->model_->nv) + 1],
        agent->states_[agent->state_]->mocap()[1], 1.0e-1);
    EXPECT_NEAR(
        agent->ActivePlanner().BestTrajectory()->states
            [(agent->steps_ - 1) * (agent->model_->nq + agent->model_->nv) + 2],
        0.0, 1.0e-1);
    EXPECT_NEAR(
        agent->ActivePlanner().BestTrajectory()->states
            [(agent->steps_ - 1) * (agent->model_->nq + agent->model_->nv) + 3],
        0.0, 1.0e-1);

    // delete data
    mj_deleteData(data);

    // delete model
    mj_deleteModel(model);
  }
};

TEST_F(AgentTest, Initialization) { TestInitialization(); }

TEST_F(AgentTest, Plan) { TestPlan(); }

}  // namespace mjpc
