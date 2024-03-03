# %%
# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import mujoco
import direct_optimizer
import numpy as np
from numpy import typing as npt
import pathlib

# %%
## Example
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../mjpc/test/testdata/estimator/particle/task1D_framepos.xml"
)
# create simulation model + data
model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

# %%
# rollout
T = 10
data.qpos[0] = 1.0
qpos = []
sensor = []
for t in range(T):
  mujoco.mj_forward(model, data)
  qpos.append(data.qpos)
  sensor.append(data.sensordata)
  mujoco.mj_Euler(model, data)


# %%
# update site pos parameters
def parameter_update(model: mujoco.MjModel, parameter: npt.ArrayLike):
  model.site_sameframe[0] = 0
  model.site_sameframe[1] = 0

  # set site 0 position
  model.site_pos[0][0] = parameter[0]
  model.site_pos[0][1] = parameter[1]
  model.site_pos[0][2] = parameter[2]

  # set site 1 position
  model.site_pos[1][0] = parameter[3]
  model.site_pos[1][1] = parameter[4]
  model.site_pos[1][2] = parameter[5]


# set up optimizer
optimizer = direct_optimizer.DirectOptimizer(
    model, T, num_parameter=6, parameter_update=parameter_update
)

# settings
optimizer.max_iterations = 1000
optimizer.max_search_iterations = 1000

# initialize qpos, sensor targets
for t in range(T):
  optimizer.qpos[:, t] = qpos[t]
  optimizer.sensor_target[:, t] = sensor[t]

# weights
optimizer.weights_force[:] = 1000.0
optimizer.weights_sensor[:] = 1000.0

# initialize parameters
optimizer.parameter_target = np.hstack([model.site_pos[0], model.site_pos[1]])
optimizer.parameter = np.copy(optimizer.parameter_target)
optimizer.parameter[2] += 1.0  # perturb
optimizer.parameter[5] -= 1.0  # perturb
optimizer.weight_parameter = 0.0  # don't need to parameter cost

print("initialized:")
print(" parameter: ", optimizer.parameter)
print(" target: ", optimizer.parameter_target)
# %%
# optimize
optimizer.optimize()

# parameter
print("optimized:")
print(" parameter: ", optimizer.parameter)
print(" target: ", optimizer.parameter_target)

# status
optimizer.status()
