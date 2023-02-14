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

#include "mjpc/tasks/tasks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include "mjpc/tasks/acrobot/acrobot.h"
#include "mjpc/tasks/cartpole/cartpole.h"
#include "mjpc/tasks/hand/hand.h"
#include "mjpc/tasks/humanoid/stand/stand.h"
#include "mjpc/tasks/humanoid/tracking/tracking.h"
#include "mjpc/tasks/humanoid/walk/walk.h"
#include "mjpc/tasks/panda/panda.h"
// DEEPMIND INTERNAL IMPORT
#include "mjpc/tasks/mts/parameterized.h"
#include "mjpc/tasks/particle/particle.h"
#include "mjpc/tasks/quadrotor/quadrotor.h"
#include "mjpc/tasks/quadruped/quadruped.h"
#include "mjpc/tasks/swimmer/swimmer.h"
#include "mjpc/tasks/walker/walker.h"

namespace mjpc {

std::vector<std::shared_ptr<Task>> GetTasks() {
  std::vector<std::shared_ptr<Task>> taskVector;
  taskVector.push_back(std::make_shared<Acrobot>());
  taskVector.push_back(std::make_shared<Cartpole>());
  taskVector.push_back(std::make_shared<Hand>());
  taskVector.push_back(std::make_shared<humanoid::Stand>());
  taskVector.push_back(std::make_shared<humanoid::Tracking>());
  taskVector.push_back(std::make_shared<humanoid::Walk>());
// DEEPMIND INTERNAL TASKS
  taskVector.push_back(std::make_shared<Panda>());
  taskVector.push_back(std::make_shared<Particle>());
  taskVector.push_back(std::make_shared<Quadrotor>());
  taskVector.push_back(std::make_shared<QuadrupedFlat>());
  taskVector.push_back(std::make_shared<QuadrupedHill>());
  taskVector.push_back(std::make_shared<Swimmer>());
  taskVector.push_back(std::make_shared<Walker>());
  return taskVector;
}
}  // namespace mjpc
