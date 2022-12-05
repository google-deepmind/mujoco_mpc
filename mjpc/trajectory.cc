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

#include "trajectory.h"

#include <algorithm>
#include <functional>
#include <iostream>

#include <mujoco/mujoco.h>
#include "utilities.h"

namespace mjpc {
namespace {
// maximum return value
inline constexpr double kMaxReturnValue = 1.0e6;
}

// initialize dimensions
void Trajectory::Initialize(int dim_state, int dim_action, int dim_residual,
                            int horizon) {
  this->horizon = horizon;
  this->dim_state = dim_state;
  this->dim_action = dim_action;
  this->dim_feature = dim_residual;
}

// allocate memory
void Trajectory::Allocate(int T) {
  // states
  states.resize(dim_state * T);

  // actions
  actions.resize(dim_action * T);

  // costs
  costs.resize(T);

  // residual
  residual.resize(dim_feature * T);

  // times
  times.resize(T);

  // traces
  trace.resize(3 * T);
}

// reset memory to zeros
void Trajectory::Reset(int T) {
  // states
  std::fill(states.begin(), states.begin() + dim_state * T, 0.0);

  // actions
  std::fill(actions.begin(), actions.begin() + dim_action * T, 0.0);

  // times
  std::fill(times.begin(), times.begin() + T, 0.0);

  // costs
  std::fill(costs.begin(), costs.begin() + T, 0.0);
  std::fill(residual.begin(), residual.begin() + dim_feature * T, 0.0);
  total_return = 0.0;

  // traces
  std::fill(trace.begin(), trace.begin() + 3 * T, 0.0);
}

// simulate model forward in time with continuous time policy
void Trajectory::Rollout(
    std::function<void(double* action, const double* state, double time)>
        policy,
    const Task* task, const mjModel* model, mjData* data, const double* state,
    double time, const double* mocap, int steps) {
  // horizon
  horizon = steps;

  // set initial state
  mju_copy(states.data(), state, dim_state);
  mju_copy(data->qpos, state, model->nq);
  mju_copy(data->qvel, state + model->nq, model->nv);
  mju_copy(data->act, state + model->nq + model->nv, model->na);

  // set initial time
  times[0] = time;
  data->time = time;

  // set mocap
  for (int i = 0; i < model->nmocap; i++) {
    mju_copy(data->mocap_pos + 3 * i, mocap + 7 * i, 3);
    mju_copy(data->mocap_quat + 4 * i, mocap + 7 * i + 3, 4);
  }

  // action from policy
  policy(actions.data(), states.data(), time);
  mju_copy(data->ctrl, actions.data(), model->nu);

  // initial residual
  task->Residuals(model, data, residual.data());

  // initial trace
  double* tr = SensorByName(model, data, "trace");
  if (tr) mju_copy(trace.data(), tr, 3);

   // step simulation
  mj_step(model, data);

  // check for step warnings
  if (CheckWarnings(data)) {
    total_return = kMaxReturnValue;
    std::cerr << "Rollout divergence at step\n";
    return;
  }

  for (int t = 1; t < horizon - 1; t++) {
    // record state
    mju_copy(DataAt(states, t * dim_state), data->qpos, model->nq);
    mju_copy(DataAt(states, t * dim_state + model->nq), data->qvel, model->nv);
    mju_copy(DataAt(states, t * dim_state + model->nq + model->nv), data->act,
             model->na);
    times[t] = data->time;

    // set action
    policy(DataAt(actions, t * model->nu), DataAt(states, t * dim_state),
           data->time);
    mju_copy(data->ctrl, DataAt(actions, t * model->nu), model->nu);

    // residual
    task->Residuals(model, data, DataAt(residual, t * dim_feature));

    // record trace
    tr = SensorByName(model, data, "trace");
    if (tr) mju_copy(DataAt(trace, t * 3), tr, 3);

    // step simulation
    mj_step(model, data);

    // check for step warnings
    if (CheckWarnings(data)) {
      total_return = kMaxReturnValue;
      std::cerr << "Rollout divergence\n";
      return;
    }
  }

  // record final state
  mju_copy(DataAt(states, (horizon - 1) * dim_state), data->qpos, model->nq);
  mju_copy(DataAt(states, (horizon - 1) * dim_state + model->nq), data->qvel,
           model->nv);
  mju_copy(DataAt(states, (horizon - 1) * dim_state + model->nq + model->nv),
           data->act, model->na);
  times[horizon - 1] = data->time;

  // copy final action
  if (horizon > 1) {
    mju_copy(DataAt(actions, (horizon - 1) * dim_action),
             DataAt(actions, (horizon - 2) * dim_action), dim_action);
  } else {
    mju_zero(DataAt(actions, (horizon - 1) * dim_action), dim_action);
  }

  // final residual
  task->Residuals(model, data, DataAt(residual, (horizon - 1) * dim_feature));

  // final trace
  tr = SensorByName(model, data, "trace");
  if (tr) mju_copy(DataAt(trace, (horizon - 1) * 3), tr, 3);

  // check for step warnings
  if (CheckWarnings(data)) {
    total_return = kMaxReturnValue;
    std::cerr << "Rollout divergence\n";
    return;
  }

  // compute return
  UpdateReturn(task);
}

// simulate model forward in time with discrete-time indexed policy
void Trajectory::RolloutDiscrete(
    std::function<void(double* action, const double* state, int index)>
        policy,
    const Task* task, const mjModel* model, mjData* data, const double* state,
    double time, const double* mocap, int steps) {
  // horizon
  horizon = steps;

  // set initial state
  mju_copy(states.data(), state, dim_state);
  mju_copy(data->qpos, state, model->nq);
  mju_copy(data->qvel, state + model->nq, model->nv);
  mju_copy(data->act, state + model->nq + model->nv, model->na);

  // set initial time
  times[0] = time;
  data->time = time;

  // set mocap
  for (int i = 0; i < model->nmocap; i++) {
    mju_copy(data->mocap_pos + 3 * i, mocap + 7 * i, 3);
    mju_copy(data->mocap_quat + 4 * i, mocap + 7 * i + 3, 4);
  }

  // action from policy
  policy(actions.data(), states.data(), 0);
  mju_copy(data->ctrl, actions.data(), model->nu);

  // initial residual
  task->Residuals(model, data, residual.data());

  // initial trace
  double* tr = SensorByName(model, data, "trace");
  if (tr) mju_copy(trace.data(), tr, 3);

  // step simulation
  mj_step(model, data);

  // check for step warnings
  if (CheckWarnings(data)) {
    total_return = kMaxReturnValue;
    std::cerr << "Rollout divergence at step\n";
    return;
  }

  for (int t = 1; t < horizon - 1; t++) {
    // record state
    mju_copy(DataAt(states, t * dim_state), data->qpos, model->nq);
    mju_copy(DataAt(states, t * dim_state + model->nq), data->qvel, model->nv);
    mju_copy(DataAt(states, t * dim_state + model->nq + model->nv), data->act,
             model->na);
    times[t] = data->time;

    // set action
    policy(DataAt(actions, t * model->nu), DataAt(states, t * dim_state),
           t);
    mju_copy(data->ctrl, DataAt(actions, t * model->nu), model->nu);

    // residual
    task->Residuals(model, data, DataAt(residual, t * dim_feature));

    // record trace
    tr = SensorByName(model, data, "trace");
    if (tr) mju_copy(DataAt(trace, t * 3), tr, 3);

    // step simulation
    mj_step(model, data);

    // check for step warnings
    if (CheckWarnings(data)) {
      total_return = kMaxReturnValue;
      std::cerr << "Rollout divergence\n";
      return;
    }
  }

  // record final state
  mju_copy(DataAt(states, (horizon - 1) * dim_state), data->qpos, model->nq);
  mju_copy(DataAt(states, (horizon - 1) * dim_state + model->nq), data->qvel,
           model->nv);
  mju_copy(DataAt(states, (horizon - 1) * dim_state + model->nq + model->nv),
           data->act, model->na);
  times[horizon - 1] = data->time;

  // copy final action
  if (horizon > 1) {
    mju_copy(DataAt(actions, (horizon - 1) * dim_action),
             DataAt(actions, (horizon - 2) * dim_action), dim_action);
  } else {
    mju_zero(DataAt(actions, (horizon - 1) * dim_action), dim_action);
  }

  // final residual
  task->Residuals(model, data, DataAt(residual, (horizon - 1) * dim_feature));

  // final trace
  tr = SensorByName(model, data, "trace");
  if (tr) mju_copy(DataAt(trace, (horizon - 1) * 3), tr, 3);

  // check for step warnings
  if (CheckWarnings(data)) {
    total_return = kMaxReturnValue;
    std::cerr << "Rollout divergence\n";
    return;
  }

  // compute return
  UpdateReturn(task);
}

// calculates total_return and costs
void Trajectory::UpdateReturn(const Task* task) {
  // reset
  total_return = 0;

  for (int t = 0; t < horizon; t++) {
    // compute stage cost
    costs[t] = task->CostValue(DataAt(residual, t * task->num_residual));

    // update total return
    total_return += costs[t];
  }

  // normalize return by trajectory horizon
  total_return /= mju_max(horizon, 1);
}

}  // namespace mjpc
