// Copyright 2021 DeepMind Technologies Limited
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

#include "mjpc/app.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <absl/flags/flag.h>
#include <mujoco/mujoco.h>
#include <glfw_adapter.h>
#include "mjpc/array_safety.h"
#include "mjpc/agent.h"
#include "mjpc/estimators/estimator.h"
#include "mjpc/simulate.h"  // mjpc fork
#include "mjpc/task.h"
#include "mjpc/threadpool.h"
#include "mjpc/utilities.h"

ABSL_FLAG(bool, planner_enabled, false,
          "If true, the planner will run on startup");
ABSL_FLAG(float, sim_percent_realtime, 100,
          "The realtime percentage at which the simulation will be launched.");
ABSL_FLAG(bool, estimator_enabled, false,
          "If true, estimator loop will run on startup");
ABSL_FLAG(bool, show_left_ui, true,
          "If true, the left UI (ui0) will be visible on startup");
ABSL_FLAG(bool, show_plot, true,
          "If true, the plots will be visible on startup");
ABSL_FLAG(bool, show_info, true,
          "If true, the infotext panel will be visible on startup");


namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::util_mjpc;

// maximum mis-alignment before re-sync (simulation seconds)
const double syncMisalign = 0.1;

// fraction of refresh available for simulation
const double simRefreshFraction = 0.7;

// model and data
mjModel* m = nullptr;
mjData* d = nullptr;

// control noise variables
mjtNum* ctrlnoise = nullptr;

using Seconds = std::chrono::duration<double>;

// --------------------------------- callbacks ---------------------------------
std::unique_ptr<mj::Simulate> sim;

// controller
extern "C" {
void controller(const mjModel* m, mjData* d);
}

// controller callback
void controller(const mjModel* m, mjData* data) {
  // if agent, skip
  if (data != d) {
    return;
  }
  // if simulation:
  if (sim->agent->action_enabled) {
    sim->agent->ActivePlanner().ActionFromPolicy(
        data->ctrl, &sim->agent->state.state()[0],
        sim->agent->state.time());
  }
  // if noise
  if (!sim->agent->allocate_enabled && sim->uiloadrequest.load() == 0 &&
      sim->ctrl_noise_std) {
    for (int j = 0; j < sim->m->nu; j++) {
      data->ctrl[j] += ctrlnoise[j];
    }
  }
}

// sensor
extern "C" {
void sensor(const mjModel* m, mjData* d, int stage);
}

// sensor callback
void sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC) {
    if (!sim->agent->allocate_enabled && sim->uiloadrequest.load() == 0) {
      if (sim->agent->IsPlanningModel(model)) {
        // the planning thread and rollout threads don't need
        // synchronization when using PlanningResidual.
        const mjpc::ResidualFn* residual = sim->agent->PlanningResidual();
        residual->Residual(model, data, data->sensordata);
      } else {
        // this residual is used by the physics thread and the UI thread (for
        // plots), and is run with a shared lock, to safely run with changes to
        // weights and parameters
        sim->agent->ActiveTask()->Residual(model, data, data->sensordata);
      }
    }
  }
}

//--------------------------------- simulation ---------------------------------

mjModel* LoadModel(const mjpc::Agent* agent, mj::Simulate& sim) {
  mjpc::Agent::LoadModelResult load_model = sim.agent->LoadModel();
  mjModel* mnew = load_model.model.release();
  mju::strcpy_arr(sim.load_error, load_model.error.c_str());

  if (!mnew) {
    std::cout << load_model.error << "\n";
    return nullptr;
  }

  // compiler warning: print and pause
  if (!load_model.error.empty()) {
    std::cout << "Model compiled, but simulation warning (paused):\n  "
              << load_model.error << "\n";
    sim.run = 0;
  }

  return mnew;
}

// estimator in background thread
void EstimatorLoop(mj::Simulate& sim) {
  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.uiloadrequest.load() == 0) {
      // estimator
      int active_estimator = sim.agent->ActiveEstimatorIndex();
      mjpc::Estimator* estimator = &sim.agent->ActiveEstimator();

      // estimator update
      if (!active_estimator) {
        std::this_thread::yield();
        continue;
      } else {
        // start timer
        auto start = std::chrono::steady_clock::now();

        // set values from GUI
        estimator->SetGUIData();

        // get simulation state (lock physics thread)
        {
          const std::lock_guard<std::mutex> lock(sim.mtx);
          // copy simulation ctrl
          mju_copy(sim.agent->ctrl.data(), d->ctrl, m->nu);

          // copy simulation sensor
          mju_copy(sim.agent->sensor.data(), d->sensordata, m->nsensordata);

          // copy simulation time
          estimator->Data()->time = d->time;

          // copy simulation mocap
          mju_copy(estimator->Data()->mocap_pos, d->mocap_pos, 3 * m->nmocap);
          mju_copy(estimator->Data()->mocap_quat, d->mocap_quat, 4 * m->nmocap);

          // copy simulation userdata
          mju_copy(estimator->Data()->userdata, d->userdata, m->nuserdata);
        }

        // update filter using latest ctrl and sensor copied from physics thread
        estimator->Update(sim.agent->ctrl.data(), sim.agent->sensor.data());

        // estimator state to planner
        double* state = estimator->State();
        sim.agent->state.Set(m, state, state + m->nq, state + m->nq + m->nv,
                             d->mocap_pos, d->mocap_quat, d->userdata, d->time);

        // wait (us)
        // TODO(taylor): confirm valid for slowdown
        while (mjpc::GetDuration(start) <
               1.0e6 * estimator->Model()->opt.timestep) {
        }
      }
    }
  }
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim) {
  // cpu-sim synchronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      // TODO(nimrod): Implement drag and drop support in MJPC
    }

    // ----- task reload ----- //
    if (sim.uiloadrequest.load() == 1) {
      // get new model + task
      sim.filename = sim.agent->GetTaskXmlPath(sim.agent->gui_task_id);

      mjModel* mnew = LoadModel(sim.agent.get(), sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.agent->Initialize(mnew);
        sim.agent->plot_enabled = absl::GetFlag(FLAGS_show_plot);
        sim.agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);
        sim.agent->Allocate();

        // set home keyframe
        int home_id = mj_name2id(mnew, mjOBJ_KEY, "home");
        if (home_id >= 0) {
          mj_resetDataKeyframe(mnew, dnew, home_id);
          sim.agent->Reset(dnew->ctrl);
        } else {
          sim.agent->Reset();
        }
        sim.agent->PlotInitialize();

        sim.Load(mnew, dnew, sim.filename, true);
        m = mnew;
        d = dnew;
        mj_forward(m, d);

        // allocate ctrlnoise
        free(ctrlnoise);
        ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
        mju_zero(ctrlnoise, m->nu);
      }

      // decrement counter
      sim.uiloadrequest.fetch_sub(1);
    }

    // reload GUI
    if (sim.uiloadrequest.load() == -1) {
      sim.Load(sim.m, sim.d, sim.filename.c_str(), false);
      sim.uiloadrequest.fetch_add(1);
    }
    // ----------------------- //

    // sleep for 1 ms or yield, to let main thread run
    //  yield results in busy wait - which has better timing but kills battery
    //  life
    if (sim.run && sim.busywait) {
      std::this_thread::yield();
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }

    {
      // lock the sim mutex
      const std::lock_guard<std::mutex> lock(sim.mtx);

      if (m) {  // run only if model is present
        sim.agent->ActiveTask()->Transition(m, d);

        // running
        if (sim.run) {
          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // inject noise
          if (sim.ctrl_noise_std) {
            // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
            mjtNum rate = mju_exp(-m->opt.timestep / sim.ctrl_noise_rate);
            mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

            for (int i = 0; i < m->nu; i++) {
              // update noise
              ctrlnoise[i] =
                  rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

              // noise added in controller callback
            }
          }

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

          // misalignment condition: distance from target sim time is bigger
          // than maximum misalignment `syncMisalign`
          bool misaligned = mju_abs(Seconds(elapsedCPU).count() / slowdown -
                                    elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 ||
              syncCPU.time_since_epoch().count() == 0 || misaligned ||
              sim.speed_changed) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speed_changed = false;

            // clear old perturbations, apply new
            mju_zero(d->xfrc_applied, 6 * m->nbody);
            sim.ApplyPosePerturbations(0);  // move mocap bodies only
            sim.ApplyForcePerturbations();

            // run single step, let next iteration deal with timing
            sim.agent->ExecuteAllRunBeforeStepJobs(m, d);
            mj_step(m, d);
          } else {  // in-sync: step until ahead of cpu
            bool measured = false;
            mjtNum prevSim = d->time;
            double refreshTime = simRefreshFraction / sim.refresh_rate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim) * slowdown) <
                       mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU <
                       Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measured_slowdown =
                    std::chrono::duration<double>(elapsedCPU).count() /
                    elapsedSim;
                measured = true;
              }

              // clear old perturbations, apply new
              mju_zero(d->xfrc_applied, 6 * m->nbody);
              sim.ApplyPosePerturbations(0);  // move mocap bodies only
              sim.ApplyForcePerturbations();

              // call mj_step
              sim.agent->ExecuteAllRunBeforeStepJobs(m, d);
              mj_step(m, d);

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }
        } else {  // paused
          // apply pose perturbation
          sim.ApplyPosePerturbations(1);  // move mocap and dynamic bodies

          // still accept jobs when simulation is paused
          sim.agent->ExecuteAllRunBeforeStepJobs(m, d);

          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
          sim.speed_changed = true;
        }
      }
    }  // release sim.mtx

    // state
    if (sim.uiloadrequest.load() == 0) {
      // set ground truth state if no active estimator
      if (!sim.agent->ActiveEstimatorIndex() || !sim.agent->estimator_enabled) {
        sim.agent->state.Set(m, d);
      }
    }
  }
}
}  // namespace

// ------------------------------- main ----------------------------------------

namespace mjpc {

MjpcApp::MjpcApp(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {
  // MJPC
  printf("MuJoCo MPC (MJPC)\n");

  // MuJoCo
  std::printf(" MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have Different versions");
  }

  // threads
  printf(" Hardware threads:  %i\n", mjpc::NumAvailableHardwareThreads());

  if (sim != nullptr) {
    mju_error("Multiple instances of MjpcApp created.");
    return;
  }
  sim = std::make_unique<mj::Simulate>(
      std::make_unique<mujoco::GlfwAdapter>(),
      std::make_shared<Agent>());

  sim->agent->SetTaskList(std::move(tasks));
  sim->agent->gui_task_id = task_id;

  sim->filename = sim->agent->GetTaskXmlPath(sim->agent->gui_task_id);
  m = LoadModel(sim->agent.get(), *sim);
  if (m) d = mj_makeData(m);

  // set home keyframe
  int home_id = mj_name2id(m, mjOBJ_KEY, "home");
  if (home_id >= 0) mj_resetDataKeyframe(m, d, home_id);

  sim->mnew = m;
  sim->dnew = d;

  // control noise
  free(ctrlnoise);
  ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
  mju_zero(ctrlnoise, m->nu);

  // agent
  sim->agent->estimator_enabled = absl::GetFlag(FLAGS_estimator_enabled);
  sim->agent->Initialize(m);
  sim->agent->Allocate();
  sim->agent->Reset();
  sim->agent->PlotInitialize();

  sim->agent->plan_enabled = absl::GetFlag(FLAGS_planner_enabled);

  // Get the index of the closest sim percentage to the input.
  float desired_percent = absl::GetFlag(FLAGS_sim_percent_realtime);
  auto closest = std::min_element(
      std::begin(sim->percentRealTime), std::end(sim->percentRealTime),
      [&](float a, float b) {
        return std::abs(a - desired_percent) < std::abs(b - desired_percent);
      });
  sim->real_time_index =
      std::distance(std::begin(sim->percentRealTime), closest);

  sim->delete_old_m_d = true;
  sim->loadrequest = 2;

  sim->ui0_enable = absl::GetFlag(FLAGS_show_left_ui);
  sim->info = absl::GetFlag(FLAGS_show_info);
}

MjpcApp::~MjpcApp() {
  sim.reset();
}

// run event loop
void MjpcApp::Start() {
  // threads
  printf("  physics        :  %i\n", 1);
  printf("  render         :  %i\n", 1);
  printf("  Planner        :  %i\n", 1);
  printf("    planning     :  %i\n", sim->agent->planner_threads());
  printf("  Estimator      :  %i\n", sim->agent->estimator_threads());
  printf("    estimation   :  %i\n", sim->agent->estimator_enabled);

  // set control callback
  mjcb_control = controller;

  // set sensor callback
  mjcb_sensor = sensor;

  // one-off preparation:
  sim->InitializeRenderLoop();

  // start physics thread
  mjpc::ThreadPool physics_pool(1);
  physics_pool.Schedule([]() { PhysicsLoop(*sim); });

  // start estimator thread
  mjpc::ThreadPool estimator_pool(1);
  if (sim->agent->estimator_enabled) {
    estimator_pool.Schedule([]() { EstimatorLoop(*sim); });
  }

  {
    // start plan thread
    mjpc::ThreadPool plan_pool(1);
    plan_pool.Schedule(
        []() { sim->agent->Plan(sim->exitrequest, sim->uiloadrequest); });

    // now that planning was forked, the main thread can render

    // start simulation UI loop (blocking call)
    sim->RenderLoop();
  }
}

mj::Simulate* MjpcApp::Sim() {
  return sim.get();
}

void StartApp(std::vector<std::shared_ptr<mjpc::Task>> tasks, int task_id) {
  MjpcApp app(std::move(tasks), task_id);
  app.Start();
}

}  // namespace mjpc
