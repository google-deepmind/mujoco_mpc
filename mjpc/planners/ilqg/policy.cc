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

#include "mjpc/planners/ilqg/policy.h"

#include <algorithm>

#include <mujoco/mujoco.h>
#include "mjpc/task.h"
#include "mjpc/trajectory.h"
#include "mjpc/utilities.h"

namespace mjpc {

// allocate memory
void iLQGPolicy::Allocate(const mjModel* model, const Task& task, int horizon) {
  // model
  this->model = model;

  // reference trajectory
  trajectory.Initialize(model->nq + model->nv + model->na, model->nu,
                        task.num_residual, task.num_trace,
                        kMaxTrajectoryHorizon);
  trajectory.Allocate(kMaxTrajectoryHorizon);

  // feedback gains
  feedback_gain.resize(model->nu * (2 * model->nv + model->na) *
                       kMaxTrajectoryHorizon);

  // action improvement
  action_improvement.resize(model->nu * kMaxTrajectoryHorizon);

  // scratch
  state_scratch.resize(model->nq + model->nv + model->na);
  action_scratch.resize(model->nu);

  // interpolation
  // feedback gains ((T - 1) * dim_action * dim_state_derivative)
  feedback_gain_scratch.resize(model->nu * (2 * model->nv + model->na));

  // state interpolation (dim_state_derivative)
  state_interp.resize(model->nq + model->nv + model->na);

  // representation
  representation = GetNumberOrDefault(1, model, "ilqg_representation");
}

// reset memory to zeros
void iLQGPolicy::Reset(int horizon) {
  trajectory.Reset(horizon);
  std::fill(
      feedback_gain.begin(),
      feedback_gain.begin() + horizon * model->nu * (2 * model->nv + model->na),
      0.0);
  std::fill(action_improvement.begin(),
            action_improvement.begin() + horizon * model->nu, 0.0);
  std::fill(state_scratch.begin(),
            state_scratch.begin() + model->nq + model->nv + model->na, 0.0);
  std::fill(action_scratch.begin(), action_scratch.begin() + model->nu, 0.0);
  std::fill(
      feedback_gain_scratch.begin(),
      feedback_gain_scratch.begin() + model->nu * (2 * model->nv + model->na),
      0.0);
  std::fill(state_interp.begin(),
            state_interp.begin() + model->nq + model->nv + model->na, 0.0);

  feedback_scaling = 1.0;
}

// set action from policy
void iLQGPolicy::Action(double* action, const double* state,
                        double time) const {
  // dimension
  int dim_state = model->nq + model->nv + model->na;
  int dim_state_derivative = 2 * model->nv + model->na;
  int dim_action = model->nu;

  // find times bounds
  int bounds[2];
  FindInterval(bounds, trajectory.times, time, trajectory.horizon);

  // interpolate
  if (bounds[0] == bounds[1] || representation == 0) {
    // action reference
    ZeroInterpolation(action, time, trajectory.times, trajectory.actions.data(),
                      model->nu, trajectory.horizon - 1);

    // state reference
    ZeroInterpolation(state_interp.data(), time, trajectory.times,
                      trajectory.states.data(), dim_state, trajectory.horizon);

    // gains
    ZeroInterpolation(feedback_gain_scratch.data(), time, trajectory.times,
                      feedback_gain.data(), dim_action * dim_state_derivative,
                      trajectory.horizon - 1);
  } else if (representation == 1) {
    // action
    LinearInterpolation(action, time, trajectory.times,
                        trajectory.actions.data(), model->nu,
                        trajectory.horizon - 1);

    // state
    LinearInterpolation(state_interp.data(), time, trajectory.times,
                        trajectory.states.data(), dim_state,
                        trajectory.horizon);

    // normalize quaternions
    mj_normalizeQuat(model, state_interp.data());

    LinearInterpolation(feedback_gain_scratch.data(), time, trajectory.times,
                        feedback_gain.data(), dim_action * dim_state_derivative,
                        trajectory.horizon - 1);
  } else if (representation == 2) {
    // action
    CubicInterpolation(action, time, trajectory.times,
                       trajectory.actions.data(), model->nu,
                       trajectory.horizon - 1);

    // state
    CubicInterpolation(state_interp.data(), time, trajectory.times,
                       trajectory.states.data(), dim_state, trajectory.horizon);

    // normalize quaternions
    mj_normalizeQuat(model, state_interp.data());

    CubicInterpolation(feedback_gain_scratch.data(), time, trajectory.times,
                       feedback_gain.data(), dim_action * dim_state_derivative,
                       trajectory.horizon - 1);
  }

  // add feedback
  if (state) {
    StateDiff(model, state_scratch.data(), state_interp.data(), state, 1.0);
    mju_mulMatVec(action_scratch.data(), feedback_gain_scratch.data(),
                  state_scratch.data(), dim_action, dim_state_derivative);
    mju_addToScl(action, action_scratch.data(), feedback_scaling, dim_action);
  }

  // clamp controls
  Clamp(action, model->actuator_ctrlrange, dim_action);
}

// copy policy
void iLQGPolicy::CopyFrom(const iLQGPolicy& policy, int horizon) {
  // reference
  trajectory = policy.trajectory;

  // feedback gains
  mju_copy(feedback_gain.data(), policy.feedback_gain.data(),
           horizon * model->nu * (2 * model->nv + model->na));

  // action improvement
  mju_copy(action_improvement.data(), policy.action_improvement.data(),
           horizon * model->nu);
}

}  // namespace mjpc
