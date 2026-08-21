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

#include "third_party/mujoco_mpc/mjpc/studio/ui.h"

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include <absl/strings/str_cat.h>
#include <absl/strings/str_split.h>
#include "third_party/dear_imgui/imgui.h"
#include "third_party/implot/implot.h"
#include <mujoco/mujoco.h>
#include "mjpc/agent.h"
#include "mjpc/estimators/include.h"
#include "mjpc/planners/cross_entropy/planner.h"
#include "mjpc/planners/gradient/planner.h"
#include "mjpc/planners/gradient/spline_mapping.h"
#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/ilqs/planner.h"
#include "mjpc/planners/include.h"
#include "mjpc/planners/planner.h"
#include "mjpc/planners/sample_gradient/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/spline/spline.h"

namespace mjpc::studio {

namespace {

// Helper for Checkbox with int*
void CheckboxInt(const char* label, int* value) {
  bool b = *value;
  if (ImGui::Checkbox(label, &b)) {
    *value = b;
  }
}

// Helper to draw a MuJoCo figure using ImPlot
void DrawFigure(const mjvFigure& fig, float height) {
  if (ImPlot::BeginPlot(fig.title, ImVec2(-1, height))) {
    ImPlot::SetupAxes(fig.xlabel, nullptr);

    if (fig.range[0][0] < fig.range[0][1]) {
      ImPlot::SetupAxisLimits(ImAxis_X1, fig.range[0][0], fig.range[0][1],
                              ImGuiCond_Always);
    }
    if (fig.range[1][0] < fig.range[1][1]) {
      ImPlot::SetupAxisLimits(ImAxis_Y1, fig.range[1][0], fig.range[1][1],
                              ImGuiCond_Always);
    }

    for (int i = 0; i < mjMAXLINE; i++) {
      if (fig.linepnt[i] > 0) {
        std::string label = fig.linename[i];
        if (label.empty()) {
          label = absl::StrCat("##line_", i);
        }

        ImPlotSpec spec(
            ImPlotProp_LineColor,
            ImVec4(fig.linergb[i][0], fig.linergb[i][1],
                   fig.linergb[i][2], 1.0f),
            ImPlotProp_Stride, (int)(2 * sizeof(float)));
        ImPlot::PlotLine(label.c_str(), &fig.linedata[i][0],
                         &fig.linedata[i][1], fig.linepnt[i], spec);
      }
    }
    ImPlot::EndPlot();
  }
}

void DrawILQGSettings(iLQGPlanner* planner) {
  ImGui::SliderInt("Rollouts##ilqg", &planner->num_rollouts_gui_, 1,
                   kMaxTrajectory);

  const char* interp_items[] = {"Zero", "Linear", "Cubic"};
  int current_interp = planner->policy.representation;
  if (ImGui::Combo("Policy Interp.##ilqg", &current_interp, interp_items, 3)) {
    planner->policy.representation = current_interp;
  }
}

void DrawSamplingSettings(SamplingPlanner* planner) {
  ImGui::SliderInt("Rollouts##sampling", &planner->num_trajectory_, 1,
                   kMaxTrajectory);

  const char* interp_items[] = {"Zero", "Linear", "Cubic"};
  int current_interp = static_cast<int>(planner->interpolation_);
  if (ImGui::Combo("Spline##sampling", &current_interp, interp_items, 3)) {
    planner->interpolation_ =
        static_cast<spline::SplineInterpolation>(current_interp);
  }

  ImGui::SliderInt("Spline Pts##sampling", &planner->policy.num_spline_points,
                   MinSamplingSplinePoints, MaxSamplingSplinePoints);

  double noise_std = planner->noise_exploration[0];
  if (ImGui::SliderScalar("Noise Std##sampling", ImGuiDataType_Double,
                          &noise_std, &MinNoiseStdDev, &MaxNoiseStdDev)) {
    planner->noise_exploration[0] = noise_std;
  }

  double noise_std2 = planner->noise_exploration[1];
  if (ImGui::SliderScalar("Noise Std2##sampling", ImGuiDataType_Double,
                          &noise_std2, &MinNoiseStdDev, &MaxNoiseStdDev)) {
    planner->noise_exploration[1] = noise_std2;
  }

  bool sliding = planner->sliding_plan_ != 0;
  if (ImGui::Checkbox("Sliding Plan##sampling", &sliding)) {
    planner->sliding_plan_ = sliding ? 1 : 0;
  }
}

void DrawGradientSettings(GradientPlanner* planner) {
  ImGui::SliderInt("Rollouts##grad", &planner->num_trajectory, 1,
                   kMaxTrajectory);

  const char* interp_items[] = {"Zero", "Linear", "Cubic"};
  int current_interp = static_cast<int>(planner->policy.representation);
  if (ImGui::Combo("Spline##grad", &current_interp, interp_items, 3)) {
    planner->policy.representation =
        static_cast<spline::SplineInterpolation>(current_interp);
  }

  ImGui::SliderInt("Spline Pts##grad", &planner->policy.num_spline_points,
                   kMinGradientSplinePoints, kMaxGradientSplinePoints);
  ImGui::SliderInt("Deriv. Skip##grad", &planner->derivative_skip_, 0, 16);
}

void DrawCrossEntropySettings(CrossEntropyPlanner* planner) {
  ImGui::SliderInt("Rollouts##cem", &planner->num_trajectory_, 1,
                   kMaxTrajectory);

  const char* interp_items[] = {"Zero", "Linear", "Cubic"};
  int current_interp = static_cast<int>(planner->interpolation_);
  if (ImGui::Combo("Spline##cem", &current_interp, interp_items, 3)) {
    planner->interpolation_ =
        static_cast<spline::SplineInterpolation>(current_interp);
  }

  ImGui::SliderInt("Spline Pts##cem", &planner->policy.num_spline_points,
                   MinSamplingSplinePoints, MaxSamplingSplinePoints);

  double std_init = planner->std_initial_;
  if (ImGui::SliderScalar("Init. Std##cem", ImGuiDataType_Double, &std_init,
                          &MinNoiseStdDev, &MaxNoiseStdDev)) {
    planner->std_initial_ = std_init;
  }

  double std_min = planner->std_min_;
  double min_std_min = 0.01;
  double max_std_min = 0.5;
  if (ImGui::SliderScalar("Min. Std##cem", ImGuiDataType_Double, &std_min,
                          &min_std_min, &max_std_min)) {
    planner->std_min_ = std_min;
  }

  double explore = planner->explore_fraction_;
  double min_explore = 0.0;
  double max_explore = 1.0;
  if (ImGui::SliderScalar("Explore##cem", ImGuiDataType_Double, &explore,
                          &min_explore, &max_explore)) {
    planner->explore_fraction_ = explore;
  }

  ImGui::SliderInt("Elite##cem", &planner->n_elite_, 2, 128);
}

void DrawSampleGradientSettings(SampleGradientPlanner* planner) {
  ImGui::SliderInt("Rollouts##sg", &planner->num_trajectory_, 2,
                   kMaxTrajectory);

  const char* interp_items[] = {"Zero", "Linear", "Cubic"};
  int current_interp = static_cast<int>(planner->interpolation_);
  if (ImGui::Combo("Spline##sg", &current_interp, interp_items, 3)) {
    planner->interpolation_ =
        static_cast<spline::SplineInterpolation>(current_interp);
  }

  ImGui::SliderInt("Spline Pts##sg", &planner->policy.num_spline_points,
                   MinSamplingSplinePoints, MaxSamplingSplinePoints);

  double noise_std = planner->noise_exploration;
  if (ImGui::SliderScalar("Noise Std.##sg", ImGuiDataType_Double, &noise_std,
                          &MinNoiseStdDev, &MaxNoiseStdDev)) {
    planner->noise_exploration = noise_std;
  }

  ImGui::SliderInt("Grad. Rollouts##sg", &planner->num_gradient_, 0,
                   kMaxTrajectory);

  double filter = planner->gradient_filter_;
  double min_filter = 0.0;
  double max_filter = 1.0;
  if (ImGui::SliderScalar("Grad. Filter##sg", ImGuiDataType_Double, &filter,
                          &min_filter, &max_filter)) {
    planner->gradient_filter_ = filter;
  }
}

}  // namespace

void DrawMjpcUi(mjpc::Agent* agent, const mjModel* model, mjData* data,
                std::function<void(const std::string&)> load_model_cb,
                bool agent_active) {
  if (!agent) return;

  // Task Section
  if (ImGui::CollapsingHeader("Task", ImGuiTreeNodeFlags_DefaultOpen)) {
    // Model (Task) Selector
    std::string task_names_str = agent->GetTaskNames();
    std::vector<std::string> task_names =
        absl::StrSplit(task_names_str, '\n', absl::SkipEmpty());
    std::vector<const char*> task_items;
    task_items.reserve(task_names.size());
    for (const auto& name : task_names) {
      task_items.push_back(name.c_str());
    }
    int current_task = agent->gui_task_id;
    if (ImGui::Combo("Model", &current_task, task_items.data(),
                     task_items.size())) {
      agent->plan_enabled = false;
      agent->action_enabled = false;
      agent->gui_task_id = current_task;
      load_model_cb(agent->GetTaskXmlPath(current_task));
    }

    if (!agent_active) {
      ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                         "Plugin inactive. Please select and load a model.");
    } else if (model) {
      if (ImGui::Button("Reset Task")) {
        agent->ActiveTask()->Reset(model);
      }
      ImGui::SameLine();
      CheckboxInt("Visualize Task", &agent->ActiveTask()->visualize);

      bool updated = false;

      double min_risk = -1.0;
      double max_risk = 1.0;
      if (ImGui::SliderScalar("Risk", ImGuiDataType_Double,
                              &agent->ActiveTask()->risk, &min_risk,
                              &max_risk)) {
        updated = true;
      }

      // Weights
      if (agent->ActiveTask()->num_term) {
        if (ImGui::TreeNodeEx("Weights", ImGuiTreeNodeFlags_DefaultOpen)) {
          for (int i = 0; i < agent->ActiveTask()->num_term; i++) {
            std::string name =
                absl::StrCat(model->names + model->name_sensoradr[i]);
            double* w = agent->ActiveTask()->weight.data() + i;
            double* s = model->sensor_user + i * model->nuser_sensor;
            double min_w = s[2];
            double max_w = s[3];
            if (ImGui::SliderScalar(name.c_str(), ImGuiDataType_Double, w,
                                    &min_w, &max_w)) {
              updated = true;
            }
          }
          ImGui::TreePop();
        }
      }

      // Parameters
      if (!agent->ActiveTask()->parameters.empty()) {
        if (ImGui::TreeNodeEx("Parameters", ImGuiTreeNodeFlags_DefaultOpen)) {
          int param_idx = 0;
          for (int i = 0; i < model->nnumeric; i++) {
            std::string name =
                absl::StrCat(model->names + model->name_numericadr[i]);
            if (name.rfind("residual_select_", 0) == 0) {
              // TODO(taylorhowell): Replicate selection lists if needed.
              // For now just show as slider.
              double* p = agent->ActiveTask()->parameters.data() + param_idx;
              double min_p = 0.0;
              double max_p = 1.0;
              if (model->numeric_size[i] == 3) {
                min_p = model->numeric_data[model->numeric_adr[i] + 1];
                max_p = model->numeric_data[model->numeric_adr[i] + 2];
              }
              if (ImGui::SliderScalar(name.c_str(), ImGuiDataType_Double, p,
                                      &min_p, &max_p)) {
                updated = true;
              }
              param_idx++;
            } else if (name.rfind("residual_", 0) == 0) {
              std::string label = name.substr(9);
              double* p = agent->ActiveTask()->parameters.data() + param_idx;
              double min_p = 0.0;
              double max_p = 1.0;
              if (model->numeric_size[i] == 3) {
                min_p = model->numeric_data[model->numeric_adr[i] + 1];
                max_p = model->numeric_data[model->numeric_adr[i] + 2];
              }
              if (ImGui::SliderScalar(label.c_str(), ImGuiDataType_Double, p,
                                      &min_p, &max_p)) {
                updated = true;
              }
              param_idx++;
            }
          }
          ImGui::TreePop();
        }
      }

      if (updated) {
        agent->ActiveTask()->UpdateResidual();
      }
    }
  }

  // Agent Section
  if (agent_active && model) {
    if (ImGui::CollapsingHeader("Agent", ImGuiTreeNodeFlags_DefaultOpen)) {
      ImGui::Text("Planner Time: %.3f ms (%.1f Hz)",
                  agent->ComputeTime() / 1000.0,
                  1e6 / std::max(1.0, agent->ComputeTime()));
      if (ImGui::Button("Reset Agent")) {
        agent->Reset();
      }
      ImGui::SameLine();
      CheckboxInt("Plan", &agent->plan_enabled);
      ImGui::SameLine();
      CheckboxInt("Action", &agent->action_enabled);

      CheckboxInt("Traces", &agent->visualize_enabled);

      // Planner Selector
      std::vector<std::string> planner_names =
          absl::StrSplit(kPlannerNames, '\n', absl::SkipEmpty());
      std::vector<const char*> planner_items;
      planner_items.reserve(planner_names.size());
      for (const auto& name : planner_names) {
        planner_items.push_back(name.c_str());
      }
      int current_planner = static_cast<int>(agent->GetPlannerType());
      if (ImGui::Combo("Planner", &current_planner, planner_items.data(),
                       planner_items.size())) {
        agent->SetPlannerType(static_cast<PlannerType>(current_planner));
      }

      // Estimator Selector
      if (agent->estimator_enabled) {
        std::vector<std::string> estimator_names =
            absl::StrSplit(kEstimatorNames, '\n', absl::SkipEmpty());
        std::vector<const char*> estimator_items;
        estimator_items.reserve(estimator_names.size());
        for (const auto& name : estimator_names) {
          estimator_items.push_back(name.c_str());
        }
        int current_estimator = agent->GetEstimatorType();
        if (ImGui::Combo("Estimator", &current_estimator,
                         estimator_items.data(), estimator_items.size())) {
          agent->SetEstimatorType(current_estimator, data);
        }
      }

      // Horizon Slider
      double horizon = agent->GetHorizon();
      double min_h = kMinPlanningHorizon;
      double max_h = kMaxPlanningHorizon;
      if (ImGui::SliderScalar("Horizon", ImGuiDataType_Double, &horizon, &min_h,
                              &max_h, "%.3f s")) {
        agent->SetHorizon(horizon);
      }

      // Timestep Slider
      double timestep = agent->GetTimeStep();
      double min_t = kMinTimeStep;
      double max_t = kMaxTimeStep;
      if (ImGui::SliderScalar("Timestep", ImGuiDataType_Double, &timestep,
                              &min_t, &max_t, "%.4f s")) {
        agent->SetTimeStep(timestep);
      }

      // Integrator Selector
      const char* integrator_items[] = {"Euler", "RK4", "Implicit",
                                        "Implicit Fast"};
      int current_integrator = agent->GetIntegrator();
      if (ImGui::Combo("Integrator", &current_integrator, integrator_items,
                       4)) {
        agent->SetIntegrator(current_integrator);
      }

      // Differentiable Checkbox
      bool diff = agent->GetDifferentiable();
      if (ImGui::Checkbox("Differentiable", &diff)) {
        agent->SetDifferentiable(diff);
      }

      // Planner-specific GUI
      ImGui::Separator();
      ImGui::Text("Planner Settings");

      PlannerType planner_type = agent->GetPlannerType();
      if (planner_type == PlannerType::kILQGPlanner) {
        DrawILQGSettings(static_cast<iLQGPlanner*>(&agent->ActivePlanner()));
      } else if (planner_type == PlannerType::kSamplingPlanner) {
        DrawSamplingSettings(
            static_cast<SamplingPlanner*>(&agent->ActivePlanner()));
      } else if (planner_type == PlannerType::kGradientPlanner) {
        DrawGradientSettings(
            static_cast<GradientPlanner*>(&agent->ActivePlanner()));
      } else if (planner_type == PlannerType::kCrossEntropyPlanner) {
        DrawCrossEntropySettings(
            static_cast<CrossEntropyPlanner*>(&agent->ActivePlanner()));
      } else if (planner_type == PlannerType::kSampleGradientPlanner) {
        DrawSampleGradientSettings(
            static_cast<SampleGradientPlanner*>(&agent->ActivePlanner()));
      } else if (planner_type == PlannerType::kILQSPlanner) {
        auto* planner = static_cast<iLQSPlanner*>(&agent->ActivePlanner());
        if (ImGui::TreeNodeEx("Sampling Settings",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
          DrawSamplingSettings(&planner->sampling);
          ImGui::TreePop();
        }
        if (ImGui::TreeNodeEx("iLQG Settings",
                              ImGuiTreeNodeFlags_DefaultOpen)) {
          DrawILQGSettings(&planner->ilqg);
          ImGui::TreePop();
        }
      }
    }
  }
}

void DrawMjpcPlots(mjpc::Agent* agent, const mjModel* model, mjData* data) {
  if (!agent || !model) return;
  const AgentPlots* plots = agent->GetPlots();

  float available_height = ImGui::GetContentRegionAvail().y;
  float spacing = ImGui::GetStyle().ItemSpacing.y;
  float plot_height = (available_height - 3 * spacing) / 4.0f;
  plot_height = std::max(50.0f, plot_height);

  DrawFigure(plots->cost, plot_height);
  DrawFigure(plots->action, plot_height);
  DrawFigure(plots->planner, plot_height);
  DrawFigure(plots->timer, plot_height);
}

}  // namespace mjpc::studio
