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

#include <mujoco/mujoco.h>
#include <cassert>
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/task.h"
#include "mjpc/test/testdata/particle_residual.h"
#include "mjpc/tasks/cartpole/cartpole.h"
#include "mjpc/threadpool.h"
#include "mjpc/array_safety.h"
#include "mjpc/common/EigenTypes.h"

namespace mjpc {
  namespace {

// model
    mjModel* model;

// state
    State state;

// task
//    ParticleTestTask task;
    Cartpole task;

// sensor
    extern "C" {
    void sensor(const mjModel* m, mjData* d, int stage);
    }

// sensor callback
    void sensor(const mjModel* model, mjData* data, int stage) {
      if (stage == mjSTAGE_ACC) {
        task.Residual(model, data, data->sensordata);
      }
    }



  }  // namespace
}  // namespace mjpc

mjModel* LoadTestModel(std::string_view path) {
  // filename
  char filename[1024];
//  const std::string path_str = absl::StrCat("../../../mjpc/test/testdata/", path);;
  const std::string path_str = absl::StrCat("", path);
  mujoco::util_mjpc::strcpy_arr(filename, path_str.c_str());

  // load model
  char loadError[1024] = "";
  mjModel* model = mj_loadXML(filename, nullptr, loadError, 1000);
  if (loadError[0]) std::cerr << "load error: " << loadError << '\n';

  return model;
}

// test iLQG planner on particle task
int main() {
  using namespace mjpc;

  // load model
//  model = LoadTestModel("particle_task.xml");
  model = LoadTestModel("/home/gaussian/cmu_ri_phd/phd_research/third_party/mujoco_mpc/mjpc/tasks/cartpole/task.xml");
  task.Reset(model);

  // create data
  mjData* data = mj_makeData(model);

  // set data
  mj_forward(model, data);

  // ----- state ----- //
  // State state;
  state.Initialize(model);
  state.Allocate(model);
  state.Reset();
  state.Set(model, data);

  // ----- iLQG planner ----- //
  iLQGPlanner planner;
  planner.Initialize(model, task);
  planner.Allocate();
  planner.Reset(kMaxTrajectoryHorizon);

  // ----- iLQG planner WARM ----- //
  iLQGPlanner warm_planner;
  warm_planner.Initialize(model, task);
  warm_planner.Allocate();
  warm_planner.Reset(kMaxTrajectoryHorizon);

  // ----- settings ----- //
  int iterations = 100;
  double horizon = 1.0;
  double timestep = 0.01;
  int steps =
      mju_max(mju_min(horizon / timestep + 1, kMaxTrajectoryHorizon), 1);
  model->opt.timestep = timestep;

  // sensor callback
  mjcb_sensor = sensor;

  // threadpool
  ThreadPool pool(1);

  // set state
  planner.SetState(state);
  warm_planner.SetState(state);

  // ---- optimize ----- //
  VecDf prev_soln(model->nq + model->nv);
  VecDf soln(model->nq + model->nv);
  prev_soln.setZero();
  for (int i = 0; i < iterations; i++) {
    planner.OptimizePolicy(steps, pool);

    std::cout << "iter: " << i << " val: ";
    for (int i=0; i<(model->nq + model->nv); ++i)
    {
      soln(i) = planner.candidate_policy[0]
          .trajectory.states[(steps - 1) * (model->nq + model->nv)+i];
      std::cout << planner.candidate_policy[0]
          .trajectory.states[(steps - 1) * (model->nq + model->nv)+i] << "\t";
    }
    std::cout << std::endl;

    if ((prev_soln-soln).norm() < 1e-3)
    {
      break;
    }
    prev_soln = soln;
  }

  // ---- warm optimize ----- //
  std::cout << std::endl;
  std::cout << "NOW WARM STARTING" << std::endl;
  std::cout << std::endl;
  warm_planner.policy = planner.policy;
  prev_soln.setZero();
  for (int i = 0; i < iterations; i++) {
    warm_planner.OptimizePolicy(steps, pool);

    std::cout << "iter: " << i << " val: ";
    for (int i=0; i<(model->nq + model->nv); ++i)
    {
      soln(i) = warm_planner.candidate_policy[0]
          .trajectory.states[(steps - 1) * (model->nq + model->nv)+i];
      std::cout << warm_planner.candidate_policy[0]
          .trajectory.states[(steps - 1) * (model->nq + model->nv)+i] << "\t";
    }
    std::cout << std::endl;

    if ((prev_soln-soln).norm() < 1e-3)
    {
      break;
    }
    prev_soln = soln;
  }

// delete data
  mj_deleteData(data);

// delete model
  mj_deleteModel(model);
}
