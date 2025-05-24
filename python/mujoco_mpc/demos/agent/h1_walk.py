# Copyright 2022 DeepMind Technologies Limited
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

import datetime as dt
import mujoco
from mujoco_mpc import agent as agent_lib
import pathlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue

def runGraph(q):
    # Plot data
    fig = plt.figure()
    axs = []
    xs = []
    ys = []

    for i in range(1, 21):
        axs.append(fig.add_subplot(5, 4, i))
        ys.append([])

    def update_plots(i, xs, ys):
        action = q.get()
        # Add x and y to lists
        xs.append(dt.datetime.now().strftime('%H:%M:%S.%f'))
        xs = xs[-20:]
        for i in range(0, len(action)):
            ys[i].append(action[i])
            ys[i] = ys[i][-20:]
            # Draw x and y lists
            axs[i].clear()
            axs[i].plot(xs, ys[i])
        return axs
    anim = animation.FuncAnimation(fig, lambda i: update_plots(i, xs, ys), frames=200, interval=1, blit=True)
    plt.show(block=True)

# Cartpole model
model_path = (
    pathlib.Path(__file__).parent.parent.parent
    / "../../build/mjpc/tasks/h1/walk/task.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))


# Run GUI
with agent_lib.Agent(
    server_binary_path=pathlib.Path(agent_lib.__file__).parent
    / "mjpc"
    / "ui_agent_server",
    task_id="H1 Walk",
    model=model,
    real_time_speed=0.5,
) as agent:
    q = Queue()
    p = Process(target=runGraph, args=(q,))
    p.start()
    while True:
        action = agent.get_action()
        q.put(action)
