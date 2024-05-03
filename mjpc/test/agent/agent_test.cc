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
#include "mjpc/planners/ilqs/planner.h"
#include "mjpc/planners/sampling/planner.h"
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
    EXPECT_NEAR(agent->horizon_, 1, 1.0e-5);
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
                agent->state.mocap()[0], 1.0e-1);
    EXPECT_NEAR(
        agent->ActivePlanner().BestTrajectory()->states
            [(agent->steps_ - 1) * (agent->model_->nq + agent->model_->nv) + 1],
        agent->state.mocap()[1], 1.0e-1);
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
                agent->state.mocap()[0], 1.0e-1);
    EXPECT_NEAR(
        agent->ActivePlanner().BestTrajectory()->states
            [(agent->steps_ - 1) * (agent->model_->nq + agent->model_->nv) + 1],
        agent->state.mocap()[1], 1.0e-1);
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

  void TestPreviousSamplingPolicy() {
    model = LoadTestModel("particle_task.xml");
    mjData* data = mj_makeData(model);
    mjcb_sensor = &SensorCallback;

    ThreadPool plan_pool(128);

    // ----- initialize agent ----- //
    agent->Initialize(model);
    agent->Allocate();
    agent->Reset();
    agent->plan_enabled = true;

    bool success = false;
    agent->planner_ = 0;  // sampling
    reinterpret_cast<SamplingPlanner*>(&agent->ActivePlanner())
        ->num_trajectory_ = 128;

    // A smaller value causes flakiness because the optimizer fails to find a
    // better solution:
    reinterpret_cast<SamplingPlanner*>(&agent->ActivePlanner())
        ->noise_exploration[0] = 0.2;
    int repeats = 10;
    for (int i = 0; i < repeats; i++) {
      agent->Reset();
      data->qpos[0] = 0;
      data->qpos[1] = 0;
      data->qvel[0] = 0;
      data->qvel[1] = 0;
      data->mocap_pos[0] = 1;
      data->mocap_pos[1] = 1;
      agent->SetState(data);

      agent->PlanIteration(&plan_pool);
      double action_sample_time = 0.15;
      double orig_action[2];
      agent->ActivePlanner().ActionFromPolicy(orig_action, NULL,
                                              action_sample_time, false);
      // change target
      data->mocap_pos[0] = -1;
      data->mocap_pos[1] = -1;
      agent->SetState(data);
      agent->PlanIteration(&plan_pool);
      double updated_action[2];
      agent->ActivePlanner().ActionFromPolicy(updated_action, NULL,
                                              action_sample_time, false);
      double prev_action[2];
      agent->ActivePlanner().ActionFromPolicy(prev_action, NULL,
                                              action_sample_time, true);
      EXPECT_EQ(orig_action[0], prev_action[0]);
      EXPECT_EQ(orig_action[1], prev_action[1]);
      // since the target is lower, the new action should be meaningfully lower
      if (orig_action[0] - 1e-3 > updated_action[0] &&
          orig_action[1] - 1e-3 > updated_action[1]) {
        success = true;
        break;
      }
    }
    EXPECT_TRUE(success);

    mj_deleteData(data);
    mj_deleteModel(model);
  }

  void TestPreviousILQGPolicy() {
    model = LoadTestModel("particle_task.xml");
    mjData* data = mj_makeData(model);
    mjcb_sensor = &SensorCallback;

    ThreadPool plan_pool(128);

    // ----- initialize agent ----- //
    agent->Initialize(model);
    agent->Allocate();
    agent->Reset();
    agent->plan_enabled = true;

    agent->planner_ = 2;  // iLQG

    agent->Reset();
    data->qpos[0] = 0;
    data->qpos[1] = 0;
    data->qvel[0] = 0;
    data->qvel[1] = 0;
    data->mocap_pos[0] = 1;
    data->mocap_pos[1] = 1;
    agent->SetState(data);

    agent->PlanIteration(&plan_pool);
    double action_sample_time = 0.1;
    double orig_action[2];
    agent->ActivePlanner().ActionFromPolicy(orig_action, NULL,
                                            action_sample_time, false);
    // change target
    data->mocap_pos[0] = -1;
    data->mocap_pos[1] = -1;
    agent->SetState(data);
    agent->PlanIteration(&plan_pool);
    double updated_action[2];
    agent->ActivePlanner().ActionFromPolicy(updated_action, NULL,
                                            action_sample_time, false);
    double prev_action[2];
    agent->ActivePlanner().ActionFromPolicy(prev_action, NULL,
                                            action_sample_time, true);
    EXPECT_EQ(orig_action[0], prev_action[0]);
    EXPECT_EQ(orig_action[1], prev_action[1]);
    // since the target is lower, the new action should be meaningfully lower
    EXPECT_GT(orig_action[0] - 1e-3, updated_action[0]);
    EXPECT_GT(orig_action[1] - 1e-3, updated_action[1]);

    mj_deleteData(data);
    mj_deleteModel(model);
  }

  void TestPreviousILQSPolicy() {
    model = LoadTestModel("particle_task.xml");
    mjData* data = mj_makeData(model);
    mjcb_sensor = &SensorCallback;

    ThreadPool plan_pool(128);

    // ----- initialize agent ----- //
    agent->Initialize(model);
    agent->Allocate();
    agent->Reset();
    agent->plan_enabled = true;

    agent->planner_ = 3;  // iLQS
    iLQSPlanner* planner =
        reinterpret_cast<iLQSPlanner*>(&agent->ActivePlanner());

    // A smaller value causes flakiness because the optimizer fails to find a
    // better solution.
    planner->sampling.noise_exploration[0] = 0.2;
    bool case_tested[4];
    for (int i = 0; i < 4; ++i) {
      case_tested[i] = false;
    }
    int repeats = 10;
    // Changing the number of sampling trajectories changes the chances that the
    // sampling planner will succeed.  We try all 4 combinations, but we need to
    // repeat due to stochasticity:
    for (int i = 0; i < repeats * 4; ++i) {
      planner->sampling.num_trajectory_ = i & 1 ? 128 : 1;

      agent->Reset();
      data->qpos[0] = 0;
      data->qpos[1] = 0;
      data->qvel[0] = 0;
      data->qvel[1] = 0;
      data->mocap_pos[0] = 1;
      data->mocap_pos[1] = 1;
      agent->SetState(data);

      agent->PlanIteration(&plan_pool);
      auto orig_policy_type = planner->active_policy;
      double action_sample_time = 0.15;
      double orig_action[2];
      agent->ActivePlanner().ActionFromPolicy(orig_action, NULL,
                                              action_sample_time, false);
      // change target
      data->mocap_pos[0] = -1;
      data->mocap_pos[1] = -1;
      agent->SetState(data);

      planner->sampling.num_trajectory_ = i & 2 ? 128 : 1;
      agent->PlanIteration(&plan_pool);
      auto second_policy_type = planner->active_policy;
      EXPECT_EQ(planner->previous_active_policy, orig_policy_type);
      double updated_action[2];
      agent->ActivePlanner().ActionFromPolicy(updated_action, NULL,
                                              action_sample_time, false);
      double prev_action[2];
      agent->ActivePlanner().ActionFromPolicy(prev_action, NULL,
                                              action_sample_time, true);
      EXPECT_EQ(orig_action[0], prev_action[0]);
      EXPECT_EQ(orig_action[1], prev_action[1]);
      // since the target is lower, the new action should be meaningfully lower
      if (orig_action[0] - 1e-3 > updated_action[0] &&
          orig_action[1] - 1e-3 > updated_action[1]) {
        // record which case was hit:
        case_tested[orig_policy_type * 2 + second_policy_type] = true;
      }
      if (case_tested[0] && case_tested[1] && case_tested[2] &&
          case_tested[3]) {  // we've hit all branches successfully
        break;
      }
    }
    for (int i = 0; i < 4; ++i) {
      EXPECT_TRUE(case_tested[i])
          << "One of the iLQS combinations didn't occur, try to increase the "
             "number of repeats.";
    }
    mj_deleteData(data);
    mj_deleteModel(model);
  }
};

TEST_F(AgentTest, Initialization) { TestInitialization(); }

TEST_F(AgentTest, Plan) { TestPlan(); }

TEST_F(AgentTest, PreviousSamplingPolicy) { TestPreviousSamplingPolicy(); }
TEST_F(AgentTest, PreviousILQGPolicy) { TestPreviousILQGPolicy(); }
TEST_F(AgentTest, PreviousILQSPolicy) { TestPreviousILQSPolicy(); }

}  // namespace mjpc
