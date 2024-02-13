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

#include "mjpc/task.h"
#include <memory>
#include <string>

#include "gtest/gtest.h"
#include <mujoco/mujoco.h>
#include "mjpc/norm.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/testspeed.h"
#include "mjpc/test/load.h"

namespace mjpc {
namespace {

class TestTask : public Task {
 public:
  TestTask() : residual_(this) {}
  std::string Name() const override { return ""; }
  std::string XmlPath() const override { return ""; }

  class ResidualFn : public BaseResidualFn {
   public:
    ResidualFn(TestTask* task) : BaseResidualFn(task) {}
    void Residual(const mjModel*, const mjData*, double*) const override {}
  };

  std::unique_ptr<mjpc::ResidualFn> ResidualLocked() const override {
    return std::make_unique<ResidualFn>(residual_);
  }
  ResidualFn* InternalResidual() override { return &residual_; }
  ResidualFn residual_;
};

// test task construction
TEST(TasksTest, Task) {
  // load model
  mjModel* model = LoadTestModel("particle_task.xml");

  // task
  TestTask task;
  task.Reset(model);

  // test task
  EXPECT_NEAR(task.risk, 1.0, 1.0e-5);
  EXPECT_EQ(task.mode, 0);
  EXPECT_EQ(task.parameters.size(), 2);
  EXPECT_NEAR(task.parameters[0], 0.05, 1.0e-5);
  EXPECT_NEAR(task.parameters[1], -0.1, 1.0e-5);

  // test cost
  EXPECT_EQ(task.num_residual, 4);
  EXPECT_EQ(task.num_term, 2);
  EXPECT_EQ(task.dim_norm_residual[0], 2);
  EXPECT_EQ(task.dim_norm_residual[1], 2);
  EXPECT_EQ(task.num_norm_parameter[0], 0);
  EXPECT_EQ(task.num_norm_parameter[1], 0);
  EXPECT_EQ(task.norm[0], NormType::kQuadratic);
  EXPECT_EQ(task.norm[1], NormType::kQuadratic);
  EXPECT_NEAR(task.weight[0], 5.0, 1.0e-5);
  EXPECT_NEAR(task.weight[1], 0.1, 1.0e-5);

  // residual
  double terms[2];
  double residual[] = {1.0e-3, 2.0e-3, 3.0e-3, 4.0e-3};

  // cost terms
  task.CostTerms(terms, residual);
  double c = 0.0;
  c += 5.0 * 0.5 * mju_dot(residual, residual, 2);
  c += 0.1 * 0.5 * mju_dot(residual + 2, residual + 2, 2);

  // test cost terms
  EXPECT_NEAR(mju_abs(mju_sum(terms, 2) - c), 0.0, 1.0e-5);

  // compute weighted cost
  task.risk = 0.2;
  double tc = task.CostValue(residual);

  // test cost
  EXPECT_NEAR(mju_abs(tc - (mju_exp(task.risk * c) - 1.0) / task.risk), 0.0,
              1.0e-5);

  // delete model
  mj_deleteModel(model);
}

TEST(StepAllTasksTest, Task) {
  auto tasks = GetTasks();
  for (auto& task : tasks) {
    double cost = SynchronousPlanningCost(
                  task->Name(), /*planner_thread_count=*/1,
                  /*steps_per_planning_iteration=*/100, /*total_time=*/0.1);
    EXPECT_GT(cost, 0) << "Task " << task->Name() << " failed.";
  }
}

}  // namespace
}  // namespace mjpc
