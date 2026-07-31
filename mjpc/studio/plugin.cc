// Copyright 2026 DeepMind Technologies Limited
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

#include "third_party/mujoco_mpc/mjpc/studio/plugin.h"

#include <atomic>
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <absl/flags/flag.h>
#include <absl/strings/match.h>

ABSL_FLAG(int, mjpc_planner_threads, -1,
          "Number of planner threads. If -1, use default calculation.");
ABSL_FLAG(std::string, task, "Quadruped Flat",
          "Which model to load on startup.");

#include "third_party/dear_imgui/imgui.h"
#include "third_party/dear_imgui/imgui_internal.h"
#include <mujoco/mujoco.h>
#include "third_party/mujoco/src/experimental/studio/launcher.h"
#include "third_party/mujoco/src/experimental/platform/ux/plugin.h"
#include "mjpc/agent.h"
#include "third_party/mujoco_mpc/mjpc/studio/ui.h"
#include "mjpc/task.h"
#include "mjpc/tasks/tasks.h"
#include "mjpc/threadpool.h"

namespace mjpc::studio {

struct MjpcPluginState {
  std::shared_ptr<mjpc::Agent> agent;
  std::unique_ptr<mjpc::ThreadPool> plan_pool;
  std::atomic<bool> exitrequest{false};
  std::atomic<int> uiloadrequest{0};
  std::thread plan_thread;
  bool model_changed = false;
  std::string loaded_model_path;

  mjModel* current_model = nullptr;
  mjData* current_data = nullptr;
  std::string pending_model_file;
  bool agent_active = false;
  std::mutex mutex;
  std::atomic<bool> physics_stepped{false};

  void StopPlanner() {
    exitrequest = true;
    if (plan_thread.joinable()) {
      plan_thread.join();
    }
    plan_pool.reset();
  }

  void StartPlanner() {
    exitrequest = false;
    plan_pool = std::make_unique<ThreadPool>(1);
    plan_thread =
        std::thread([this]() { agent->Plan(exitrequest, uiloadrequest); });
  }

  ~MjpcPluginState() { StopPlanner(); }
};

static std::unique_ptr<MjpcPluginState> g_state = nullptr;

// Controller callback
void MjpcControllerCallback(const mjModel* m, mjData* d) {
  if (!g_state) return;
  // If this is a rollout (data doesn't match main simulation data), skip.
  if (d != g_state->current_data) return;

  if (!g_state->agent_active || !g_state->agent) return;

  if (g_state->agent->action_enabled) {
    g_state->agent->ActivePlanner().ActionFromPolicy(
        d->ctrl, &g_state->agent->state.state()[0],
        g_state->agent->state.time());
  }
}

// Sensor callback
void MjpcSensorCallback(const mjModel* model, mjData* data, int stage) {
  if (!g_state || !g_state->agent_active || !g_state->agent) return;

  if (stage == mjSTAGE_ACC) {
    if (!g_state->agent->allocate_enabled &&
        g_state->uiloadrequest.load() == 0) {
      if (g_state->agent->IsPlanningModel(model)) {
        const mjpc::ResidualFn* residual = g_state->agent->PlanningResidual();
        residual->Residual(model, data, data->sensordata);
      } else {
        g_state->agent->ActiveTask()->Residual(model, data, data->sensordata);
      }
    }
  }
}

void MjpcPluginPostModelLoaded(mujoco::platform::ModelPlugin* self,
                               const mjModel* model,
                               const char* model_path) {
  auto* state = static_cast<MjpcPluginState*>(self->data);
  state->model_changed = true;
  state->loaded_model_path = model_path ? model_path : "";
}

bool MjpcModelPluginDoUpdate(mujoco::platform::ModelPlugin* self,
                             mjModel* model, mjData* data) {
  auto* state = static_cast<MjpcPluginState*>(self->data);
  state->current_model = model;
  state->current_data = data;

  if (state->model_changed || !state->agent) {
    std::vector<std::shared_ptr<Task>> tasks;
    if (!state->agent) {
      tasks = GetTasks();
    }

    state->StopPlanner();

    bool success = false;
    {
      std::lock_guard<std::mutex> lock(state->mutex);

      if (!state->agent) {
        state->agent = std::make_shared<Agent>();
        state->agent->SetTaskList(std::move(tasks));
      }

      // Check if this looks like a valid MJPC model.
      // MJPC models must have at least one sensor, and the first sensor must be
      // mjSENS_USER.
      bool valid_mjpc_model =
          model && model->nsensor > 0 && model->sensor_type[0] == mjSENS_USER;

      if (valid_mjpc_model) {
        int task_index =
            state->agent->FindTaskIndexByXmlPath(state->loaded_model_path);
        if (task_index >= 0) {
          state->agent->gui_task_id = task_index;
        }
        state->agent->Initialize(model);
        int override_threads = absl::GetFlag(FLAGS_mjpc_planner_threads);
        if (override_threads > 0) {
          state->agent->SetPlannerThreads(override_threads);
        }
        state->agent->Allocate();

        // Set home keyframe if available
        int home_id = mj_name2id(model, mjOBJ_KEY, "home");
        if (home_id >= 0) {
          mj_resetDataKeyframe(model, data, home_id);
          state->agent->Reset(data->ctrl);
        } else {
          state->agent->Reset();
        }
        state->agent->PlotInitialize();
        success = true;
      }
    }

    if (success) {
      state->StartPlanner();
      state->agent_active = true;
    } else {
      state->agent_active = false;
    }
    state->model_changed = false;
  }

  // Run transition and before step jobs (handles paused state as well)
  if (state->agent && state->agent_active) {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->agent->ActiveTask()->Transition(model, data);
    state->agent->ExecuteAllRunBeforeStepJobs(model, data);

    // Set state for planner
    state->agent->state.Set(model, data);
  }

  return false;  // Let Studio handle stepping
}

void MjpcModelPluginPreStep(mujoco::platform::ModelPlugin* self,
                            const mjModel* model, mjData* data) {
  auto* state = static_cast<MjpcPluginState*>(self->data);
  if (state && state->agent_active && state->agent) {
    state->agent->ExecuteAllRunBeforeStepJobs(model, data);
    state->physics_stepped.store(true);
  }
}

void MjpcModelPluginPostStep(mujoco::platform::ModelPlugin* self,
                             const mjModel* model, mjData* data) {
  auto* state = static_cast<MjpcPluginState*>(self->data);
  if (state && state->agent_active && state->agent) {
    state->agent->state.Set(model, data);
  }
}

void MjpcModelPluginEnhanceScene(mujoco::platform::ScenePlugin* self,
                                 const mjModel* model, mjData* data,
                                 mjvScene* scene) {
  auto* state = static_cast<MjpcPluginState*>(self->data);
  if (state && state->agent_active && state->agent) {
    std::lock_guard<std::mutex> lock(state->mutex);
    if (state->agent->ActiveTask()->visualize) {
      state->agent->ActiveTask()->ModifyScene(model, data, scene);
    }
    state->agent->ModifyScene(scene);
  }
}

const char* MjpcPluginGetModelToLoad(mujoco::platform::ModelPlugin* self,
                                     int* size, char* content_type,
                                     int content_type_size, char* model_name,
                                     int model_name_size) {
  auto* state = static_cast<MjpcPluginState*>(self->data);
  if (!state->pending_model_file.empty()) {
    state->agent_active = false;
    state->StopPlanner();

    std::strncpy(model_name, state->pending_model_file.c_str(), model_name_size);
    state->pending_model_file.clear();
    *size = 0;
    return model_name;
  }
  return nullptr;
}

void MjpcGuiPluginUpdate(mujoco::platform::GuiPlugin* self) {
  auto* state = static_cast<MjpcPluginState*>(self->data);
  if (state && state->agent && state->current_model && state->current_data) {
    static int frame_count = 0;
    static bool docked = false;
    static int focus_frames = 0;
    if (!docked && frame_count > 0) {
      ImGuiWindow* options_window = ImGui::FindWindowByName("Options");
      ImGuiWindow* mjpc_window = ImGui::FindWindowByName("MJPC");
      if (options_window && options_window->DockNode) {
        if (mjpc_window && mjpc_window->DockNode) {
          docked = true;
        } else {
          ImGuiID dock_id_options = options_window->DockNode->ID;
          ImGuiDockNode* root_node = options_window->DockNode;
          while (root_node->ParentNode) {
            root_node = root_node->ParentNode;
          }
          ImGuiID root_id = root_node->ID;

          ImGui::DockBuilderDockWindow("MJPC", dock_id_options);
          ImGui::DockBuilderFinish(root_id);

          docked = true;
          focus_frames = 5;
        }
      }
    }
    if (focus_frames > 0) {
      ImGui::SetWindowFocus("MJPC");
      focus_frames--;
    }
    frame_count++;

    bool shifted = false;
    if (state->agent_active) {
      shifted = state->physics_stepped.exchange(false);
      state->agent->Plots(state->current_data, shifted ? 1 : 0);
    }

    std::lock_guard<std::mutex> lock(state->mutex);
    DrawMjpcUi(
        state->agent.get(), state->current_model, state->current_data,
        [state](const std::string& path) { state->pending_model_file = path; },
        state->agent_active);
  }
}

void MjpcPlotsPluginUpdate(mujoco::platform::GuiPlugin* self) {
  auto* state = static_cast<MjpcPluginState*>(self->data);
  if (state && state->agent && state->current_model && state->current_data &&
      state->agent_active) {
    static int frame_count = 0;
    static bool docked = false;
    static int focus_frames = 0;
    if (!docked && frame_count > 0) {
      ImGuiWindow* inspector_window = ImGui::FindWindowByName("Inspector");
      ImGuiWindow* plots_window = ImGui::FindWindowByName("Plots");
      if (inspector_window && inspector_window->DockNode) {
        if (plots_window && plots_window->DockNode) {
          docked = true;
        } else {
          ImGui::DockBuilderDockWindow("Plots", inspector_window->DockNode->ID);
          docked = true;
          focus_frames = 5;
        }
      }
    }
    if (focus_frames > 0) {
      ImGui::SetWindowFocus("Plots");
      focus_frames--;
    }
    frame_count++;

    std::lock_guard<std::mutex> lock(state->mutex);
    DrawMjpcPlots(state->agent.get(), state->current_model,
                  state->current_data);
  }
}

void PrepareToLaunch(mujoco::studio::LauncherConfig& config) {
  if (config.model_file.empty()) {
    std::string task_name = absl::GetFlag(FLAGS_task);
    auto tasks = mjpc::GetTasks();
    int task_id = -1;
    for (int i = 0; i < tasks.size(); i++) {
      if (absl::EqualsIgnoreCase(task_name, tasks[i]->Name())) {
        task_id = i;
        break;
      }
    }
    if (task_id == -1) {
      mju_error("Invalid --task flag.");
    }

    config.model_file = tasks[task_id]->XmlPath();
  }
}

}  // namespace mjpc::studio

mjPLUGIN_LIB_INIT(mjpc_studio) {
  using mjpc::studio::g_state;
  using mjpc::studio::MjpcControllerCallback;
  using mjpc::studio::MjpcGuiPluginUpdate;
  using mjpc::studio::MjpcModelPluginDoUpdate;
  using mjpc::studio::MjpcModelPluginEnhanceScene;
  using mjpc::studio::MjpcModelPluginPreStep;
  using mjpc::studio::MjpcModelPluginPostStep;
  using mjpc::studio::MjpcPlotsPluginUpdate;
  using mjpc::studio::MjpcPluginGetModelToLoad;
  using mjpc::studio::MjpcPluginPostModelLoaded;
  using mjpc::studio::MjpcPluginState;
  using mjpc::studio::MjpcSensorCallback;

  if (g_state) {
    return;  // Already initialized
  }

  g_state = std::make_unique<MjpcPluginState>();

  // Set callbacks
  mjcb_control = MjpcControllerCallback;
  mjcb_sensor = MjpcSensorCallback;

  // Register Model Plugin
  mujoco::platform::ModelPlugin model_plugin;
  model_plugin.data = g_state.get();
  model_plugin.name = "MJPC";
  model_plugin.get_model_to_load = MjpcPluginGetModelToLoad;
  model_plugin.post_model_loaded = MjpcPluginPostModelLoaded;
  model_plugin.do_update = MjpcModelPluginDoUpdate;
  model_plugin.pre_step = MjpcModelPluginPreStep;
  model_plugin.post_step = MjpcModelPluginPostStep;
  mujoco::platform::RegisterPlugin(model_plugin);

  mujoco::platform::ScenePlugin scene_plugin;
  scene_plugin.data = g_state.get();
  scene_plugin.name = "MJPC";
  scene_plugin.enhance_scene = MjpcModelPluginEnhanceScene;
  mujoco::platform::RegisterPlugin(scene_plugin);

  // Register GUI Plugin
  mujoco::platform::GuiPlugin gui_plugin;
  gui_plugin.data = g_state.get();
  gui_plugin.name = "MJPC";
  gui_plugin.active = true;
  gui_plugin.update = MjpcGuiPluginUpdate;
  mujoco::platform::RegisterPlugin(gui_plugin);

  // Register Plots GUI Plugin
  mujoco::platform::GuiPlugin plots_plugin;
  plots_plugin.data = g_state.get();
  plots_plugin.name = "Plots";
  plots_plugin.active = true;
  plots_plugin.update = MjpcPlotsPluginUpdate;
  mujoco::platform::RegisterPlugin(plots_plugin);

  // Register Key Handlers
  mujoco::platform::KeyHandlerPlugin plan_key;
  plan_key.data = g_state.get();
  plan_key.name = "MJPC Plan Toggle";
  plan_key.key_chord = ImGuiKey_Enter;
  plan_key.on_key_pressed = [](mujoco::platform::KeyHandlerPlugin* self) {
    auto* state = static_cast<MjpcPluginState*>(self->data);
    if (state && state->agent && state->agent_active) {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->agent->plan_enabled = !state->agent->plan_enabled;
    }
  };
  mujoco::platform::RegisterPlugin(plan_key);

  mujoco::platform::KeyHandlerPlugin action_key;
  action_key.data = g_state.get();
  action_key.name = "MJPC Action Toggle";
  action_key.key_chord = ImGuiKey_Backslash;
  action_key.on_key_pressed = [](mujoco::platform::KeyHandlerPlugin* self) {
    auto* state = static_cast<MjpcPluginState*>(self->data);
    if (state && state->agent && state->agent_active) {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->agent->action_enabled = !state->agent->action_enabled;
    }
  };
  mujoco::platform::RegisterPlugin(action_key);

  mujoco::platform::KeyHandlerPlugin traces_key;
  traces_key.data = g_state.get();
  traces_key.name = "MJPC Traces Toggle";
  traces_key.key_chord = ImGuiKey_9;
  traces_key.on_key_pressed = [](mujoco::platform::KeyHandlerPlugin* self) {
    auto* state = static_cast<MjpcPluginState*>(self->data);
    if (state && state->agent && state->agent_active) {
      std::lock_guard<std::mutex> lock(state->mutex);
      state->agent->visualize_enabled = !state->agent->visualize_enabled;
    }
  };
  mujoco::platform::RegisterPlugin(traces_key);
}
