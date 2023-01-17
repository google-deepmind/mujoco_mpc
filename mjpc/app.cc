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

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/strings/match.h>
#include <mujoco/mujoco.h>
#include <glfw_adapter.h>
#include "array_safety.h"
#include "agent.h"
#include "planners/include.h"
#include "simulate.h"  // mjpc fork
#include "threadpool.h"
#include "utilities.h"

ABSL_FLAG(std::string, task, "", "Which model to load on startup.");

namespace {
namespace mj = ::mujoco;
namespace mju = ::mujoco::util_mjpc;

// maximum mis-alignment before re-sync (simulation seconds)
const double syncMisalign = 0.1;

// fraction of refresh available for simulation
const double simRefreshFraction = 0.7;

// load error string length
const int kErrorLength = 1024;

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
  if (sim->agent.action_enabled) {
    sim->agent.ActivePlanner().ActionFromPolicy(
        data->ctrl, &sim->agent.ActiveState().state()[0],
        sim->agent.ActiveState().time());
  }
  // if noise
  if (!sim->agent.allocate_enabled && sim->uiloadrequest.load() == 0 &&
      sim->ctrlnoisestd) {
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
    if (!sim->agent.allocate_enabled && sim->uiloadrequest.load() == 0) {
      // users sensors must be ordered first and sequentially
      sim->agent.task().Residuals(model, data, data->sensordata);
    }
  }
}

//--------------------------------- simulation ---------------------------------

mjModel* LoadModel(std::string filename, mj::Simulate& sim) {
  // make sure filename is not empty
  if (filename.empty()) {
    return nullptr;
  }

  // load and compile
  char loadError[kErrorLength] = "";
  mjModel* mnew = 0;
  if (absl::StrContains(filename, ".mjb")) {
    mnew = mj_loadModel(filename.c_str(), nullptr);
    if (!mnew) {
      mju::strcpy_arr(loadError, "could not load binary model");
    }
  } else {
    mnew = mj_loadXML(filename.c_str(), nullptr, loadError,
                      mj::Simulate::kMaxFilenameLength);
    // remove trailing newline character from loadError
    if (loadError[0]) {
      int error_length = mju::strlen_arr(loadError);
      if (loadError[error_length - 1] == '\n') {
        loadError[error_length - 1] = '\0';
      }
    }
  }

  mju::strcpy_arr(sim.loadError, loadError);

  if (!mnew) {
    std::printf("%s\n", loadError);
    return nullptr;
  }

  // compiler warning: print and pause
  if (loadError[0]) {
    // mj_forward() below will print the warning message
    std::printf("Model compiled, but simulation warning (paused):\n  %s\n",
                loadError);
    sim.run = 0;
  }

  return mnew;
}

// simulate in background thread (while rendering in main thread)
void PhysicsLoop(mj::Simulate& sim) {
  // cpu-sim syncronization point
  std::chrono::time_point<mj::Simulate::Clock> syncCPU;
  mjtNum syncSim = 0;

  // run until asked to exit
  while (!sim.exitrequest.load()) {
    if (sim.droploadrequest.load()) {
      mjModel* mnew = LoadModel(sim.dropfilename, sim);
      sim.droploadrequest.store(false);

      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.load(sim.dropfilename.c_str(), mnew, dnew, true);

        m = mnew;
        d = dnew;
        mj_forward(m, d);

        // allocate ctrlnoise
        free(ctrlnoise);
        ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
        mju_zero(ctrlnoise, m->nu);
      }
    }

    // ----- task reload ----- //
    if (sim.uiloadrequest.load() == 1) {
      // get new model + task
      sim.filename = sim.tasks[sim.agent.task().id].xml_path;

      mjModel* mnew = LoadModel(sim.filename, sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.load(sim.filename, mnew, dnew, true);
        m = mnew;
        d = dnew;
        mj_forward(m, d);

        // allocate ctrlnoise
        free(ctrlnoise);
        ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
        mju_zero(ctrlnoise, m->nu);
      }

      // agent
      {
        std::ostringstream concatenated_task_names;
        for (const auto& task : sim.tasks) {
          concatenated_task_names << task.name << '\n';
        }
        const auto& task = sim.tasks[sim.agent.task().id];
        sim.agent.Initialize(m, concatenated_task_names.str(),
                             mjpc::kPlannerNames, task.residual,
                             task.transition);
      }
      sim.agent.Allocate();
      sim.agent.Reset();
      sim.agent.PlotInitialize();
    }

    // reload model to refresh UI
    if (sim.uiloadrequest.load() == 1) {
      mjModel* mnew =
          LoadModel(sim.tasks[sim.agent.task().id].xml_path, sim);
      mjData* dnew = nullptr;
      if (mnew) dnew = mj_makeData(mnew);
      if (dnew) {
        sim.load(sim.filename.c_str(), mnew, dnew, true);
        m = mnew;
        d = dnew;
        mj_forward(m, d);

        // allocate ctrlnoise
        free(ctrlnoise);
        ctrlnoise = static_cast<mjtNum*>(malloc(sizeof(mjtNum) * m->nu));
        mju_zero(ctrlnoise, m->nu);
      }

      // set initial configuration via keyframe
      double* qpos_key = mjpc::KeyFrameByName(sim.mnew, sim.dnew, "home");
      if (qpos_key) {
        mju_copy(sim.dnew->qpos, qpos_key, sim.mnew->nq);
      }

      // decrement counter
      sim.uiloadrequest.fetch_sub(1);
    }

    // reload GUI
    if (sim.uiloadrequest.load() == -1) {
      sim.load(sim.filename.c_str(), sim.m, sim.d, false);
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
        if (sim.agent.task().transition_status == 1) {
          sim.agent.task().Transition(m, d);
        }

        // running
        if (sim.run) {
          // record cpu time at start of iteration
          const auto startCPU = mj::Simulate::Clock::now();

          // elapsed CPU and simulation time since last sync
          const auto elapsedCPU = startCPU - syncCPU;
          double elapsedSim = d->time - syncSim;

          // inject noise
          if (sim.ctrlnoisestd) {
            // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
            mjtNum rate = mju_exp(-m->opt.timestep / sim.ctrlnoiserate);
            mjtNum scale = sim.ctrlnoisestd * mju_sqrt(1 - rate * rate);

            for (int i = 0; i < m->nu; i++) {
              // update noise
              ctrlnoise[i] =
                  rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

              // apply noise
              // d->ctrl[i] += ctrlnoise[i]; // noise is now added in controller
              // callback
            }
          }

          // requested slow-down factor
          double slowdown = 100 / sim.percentRealTime[sim.realTimeIndex];

          // misalignment condition: distance from target sim time is bigger
          // than syncmisalign
          bool misaligned =
              mju_abs(Seconds(elapsedCPU).count()/slowdown - elapsedSim) > syncMisalign;

          // out-of-sync (for any reason): reset sync times, step
          if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
              misaligned || sim.speedChanged) {
            // re-sync
            syncCPU = startCPU;
            syncSim = d->time;
            sim.speedChanged = false;

            // clear old perturbations, apply new
            mju_zero(d->xfrc_applied, 6 * m->nbody);
            sim.applyposepertubations(0);  // move mocap bodies only
            sim.applyforceperturbations();

            // run single step, let next iteration deal with timing
            mj_step(m, d);
          } else {  // in-sync: step until ahead of cpu
            bool measured = false;
            mjtNum prevSim = d->time;
            double refreshTime = simRefreshFraction / sim.refreshRate;

            // step while sim lags behind cpu and within refreshTime
            while (Seconds((d->time - syncSim)*slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                   mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime)) {
              // measure slowdown before first step
              if (!measured && elapsedSim) {
                sim.measuredSlowdown =
                    std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                measured = true;
              }

              // clear old perturbations, apply new
              mju_zero(d->xfrc_applied, 6 * m->nbody);
              sim.applyposepertubations(0);  // move mocap bodies only
              sim.applyforceperturbations();

              // call mj_step
              mj_step(m, d);

              // break if reset
              if (d->time < prevSim) {
                break;
              }
            }
          }
        } else {  // paused
          // apply pose perturbation
          sim.applyposepertubations(1);  // move mocap and dynamic bodies

          // run mj_forward, to update rendering and joint sliders
          mj_forward(m, d);
        }
      }
    }  // release sim.mtx

    // state
    if (sim.uiloadrequest.load() == 0) {
      sim.agent.ActiveState().Set(m, d);
    }
  }
}
}  // namespace

// ------------------------------- main ----------------------------------------

namespace mjpc {

// run event loop
void StartApp(std::vector<mjpc::TaskDefinition<>> tasks) {
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version()) {
    mju_error("Headers and library have Different versions");
  }

  // threads
  printf("Hardware threads: %i\n", mjpc::NumAvailableHardwareThreads());

  sim = std::make_unique<mj::Simulate>(
      std::make_unique<mujoco::GlfwAdapter>());

  sim->tasks = std::move(tasks);
  std::string task = absl::GetFlag(FLAGS_task);
  if (!task.empty()) {
    sim->agent.task().id = sim->TaskIdByName(task);
    if (sim->agent.task().id == -1) {
      std::cerr << "Invalid --task flag: '" << task << "'. Valid values:\n";
      for (const auto& task : sim->tasks) {
        std::cerr << '\t' << task.name << '\n';
      }
      mju_error("Invalid --task flag.");
    }
  }

  sim->filename = sim->tasks[sim->agent.task().id].xml_path;
  m = LoadModel(sim->filename, *sim);
  if (m) d = mj_makeData(m);
  sim->mnew = m;
  sim->dnew = d;

  sim->delete_old_m_d = true;
  sim->loadrequest = 2;

  // control noise
  free(ctrlnoise);
  ctrlnoise = (mjtNum*)malloc(sizeof(mjtNum) * m->nu);
  mju_zero(ctrlnoise, m->nu);

  // agent
  {
    std::ostringstream concatenated_task_names;
    for (const auto& task : sim->tasks) {
      concatenated_task_names << task.name << '\n';
    }
    const auto& task = sim->tasks[sim->agent.task().id];
    sim->agent.Initialize(m, concatenated_task_names.str(),
                          mjpc::kPlannerNames, task.residual,
                          task.transition);
  }
  sim->agent.Allocate();
  sim->agent.Reset();
  sim->agent.PlotInitialize();

  // planning threads
  printf("Agent threads: %i\n", sim->agent.max_threads());

  // set initial configuration via keyframe
  double* qpos_key = mjpc::KeyFrameByName(sim->mnew, sim->dnew, "home");
  if (qpos_key) {
    mju_copy(sim->dnew->qpos, qpos_key, sim->mnew->nq);
  }

  // set control callback
  mjcb_control = controller;

  // set sensor callback
  mjcb_sensor = sensor;

  // start physics thread
  mjpc::ThreadPool physics_pool(1);
  physics_pool.Schedule([]() { PhysicsLoop(*sim.get()); });

  {
    // start plan thread
    mjpc::ThreadPool plan_pool(1);
    plan_pool.Schedule(
        []() { sim->agent.Plan(sim->exitrequest, sim->uiloadrequest); });

    // start simulation UI loop (blocking call)
    sim->renderloop();
  }

  // destroy the Simulate instance
  sim.release();
}
}  // namespace mjpc
