import copy
import datetime as dt
import matplotlib.pyplot as plt
import mediapy as media
import mujoco
import mujoco.viewer
from multiprocessing import Process, Queue
import numpy as np
import pathlib
import time

from mujoco_mpc import agent as agent_lib
from mjv_utils import Line, GeomManager
from plotting_utils import PlotManager
from slider_manager import SliderManager

def step_function(x, s=0.001, step_x=0.0, orientation=1):
    return 0.5*(np.tanh(orientation*(x-step_x)/s)+1)

def add_fw_cost(geom_manager, data):
    line = geom_manager.add_line()
    line.start = lambda: data.site('upper_torso').xpos
    line.end = lambda: data.site('upper_torso').xpos + data.sensor('torso_forward').data
    line = geom_manager.add_line()
    line.start = lambda: data.body('pelvis').xpos
    line.end = lambda: data.body('pelvis').xpos + data.sensor('pelvis_forward').data
    
    # Right foot info
    point = geom_manager.add_point()
    point.location = lambda: data.sensor('foot_right_xbody').data
    point = geom_manager.add_point(geom_rgba = [0,1,0,1])
    point.location = lambda: data.sensor('foot_right').data
    line = geom_manager.add_line()
    line.start = lambda: data.body('right_ankle_link').xpos
    line.end = lambda: data.body('right_ankle_link').xpos + data.sensor('foot_right_forward').data
    
    # Left foot info
    line = geom_manager.add_line()
    line.start = lambda: data.sensor('foot_left_xbody').data
    line.end = lambda: data.sensor('foot_left_xbody').data + data.sensor('foot_left_forward').data
    line = geom_manager.add_line(geom_rgba=[0, 1, 0, 1])
    line.start = lambda: data.sensor('foot_left_xbody').data
    line.end = lambda: data.sensor('foot_left_xbody').data + data.sensor('foot_left_left').data
    line = geom_manager.add_line(geom_rgba=[0, 0, 1, 1])
    line.start = lambda: data.sensor('foot_left_xbody').data
    line.end = lambda: data.sensor('foot_left_xbody').data + data.sensor('foot_left_up').data

def add_foot_distance(geom_manager, data):
    line = geom_manager.add_line(geom_rgba=[0, 0, 1, 1])
    line.start = lambda: data.body('right_ankle_link').xpos
    line.end = lambda: data.body('left_ankle_link').xpos

def configure_plot_manager(plot_manager: PlotManager, model):
    w = plot_manager.add_window((6, 1), "Cost function terms")
    plot_manager.add_plot(w, "Foot distance")
    plot_manager.add_plot(w, "Right foot velocity x")
    plot_manager.add_plot(w, "Right foot velocity y")
    plot_manager.add_plot(w, "Right foot velocity z")
    plot_manager.add_plot(w, "Left ankle height")
    plot_manager.add_plot(w, "Right ankle height")
    w = plot_manager.add_window((10, 2), "Errors")
    for i in range(1,model.njnt):
        plot_manager.add_plot(w, model.joint(i).name, (-0.4, 0.4))
    w = plot_manager.add_window((10, 2), "Debug")
    for i in range(1,model.njnt):
        plot_manager.add_plot(w, model.joint(i).name, (-0.4, 0.4))
        
def add_slider_foreach_joint(slider_manager, model, default_value, min_value, max_value, prefix=""):
    for i in range(1,model.njnt):
        joint = model.joint(i)
        default_value_val = default_value(i, joint) if callable(default_value) else default_value
        min_value_val = min_value(i, joint) if callable(min_value) else min_value
        max_value_val = max_value(i, joint) if callable(max_value) else max_value
        slider_manager.add_slider(prefix + joint.name, default_value_val, min_value_val, max_value_val)

def main():
    SIMULATION_RUNNING = False
    i = 0
    steps_per_planning_iteration = 10
    # kp = np.asarray([200, 200, 200, 300, 40, 200, 200, 200, 300, 40, 300, 100, 100, 100, 100, 100, 100, 100, 10])
    # kd = np.asarray([5, 5, 5, 6, 2, 5, 5, 5, 6, 2, 6, 2, 2, 2, 2, 2, 2, 2, 2])
    kp = np.asarray([150, 500, 250, 350, 250, 150, 500, 250, 350, 250, 500, 150, 50, 50, 50, 150, 50, 50, 50])
    kd = np.asarray([0, 35, 10, 10, 0, 0, 35, 10, 10, 0, 10, 5, 5, 5, 1.5, 5, 5, 5, 1.5])
    #200 5 hip
    #300 6 knee torso
    #40  2 ankle
    #100 2 shoulder1 shoulder2 elbow 
    def handle_key(key):
        nonlocal SIMULATION_RUNNING
        if key == ord(" "):
            print("SIMULATION PAUSED/RESUMED")
            SIMULATION_RUNNING = not SIMULATION_RUNNING 
    # model
    model_path = (
        pathlib.Path(__file__).parent
        / "../build/mjpc/tasks/h1/walk/task.xml"
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    model.opt.timestep = 0.002
    # data
    data = mujoco.MjData(model)
    # agent
    agent = agent_lib.Agent(task_id="H1 Walk", 
                            model=model, 
                            server_binary_path=pathlib.Path(agent_lib.__file__).parent
                            / "mjpc"
                            / "agent_server")

    # weights
    #agent.set_cost_weights({"Heigth": 0, "Velocity": 0.15})
    print("Cost weights:", agent.get_cost_weights())

    # parameters
    agent.set_task_parameter("Torso", 1.3)
    agent.set_task_parameter("Speed", 0.7)
    print("Parameters:", agent.get_task_parameters())

    # rollout
    mujoco.mj_resetData(model, data)
    
    # mocap
    mocap_path = [np.asarray([2.0, 2.0, 0.25]),
                  np.asarray([2.0, -2.0, 0.25]), 
                  np.asarray([-2.0, -2.0, 0.25]), 
                  np.asarray([-2.0, 2.0, 0.25])]
    current_mocap = 0
    data.mocap_pos[0] = mocap_path[current_mocap]
    
    
    # simulate
    with mujoco.viewer.launch_passive(model, data, key_callback=handle_key) as viewer:
        start = time.time()
        
        geom_manager = GeomManager(viewer.user_scn)
        add_fw_cost(geom_manager, data)
        add_foot_distance(geom_manager, data)
        
        plot_manager = PlotManager()
        configure_plot_manager(plot_manager, model)
        plot_manager.start()
        
        kp_slider_manager = SliderManager(name = "kp")
        kd_slider_manager = SliderManager(name = "kd")
        ref_slider_manager = SliderManager(name = "ref")
        add_slider_foreach_joint(kp_slider_manager, model, lambda i, j: kp[i-1], 0.0, 800.0)
        add_slider_foreach_joint(kd_slider_manager, model, lambda i, j: kd[i-1], 0.0, 50.0)
        add_slider_foreach_joint(ref_slider_manager, model, lambda _, j: j.qpos0, lambda _, j: j.range[0], lambda _, j: j.range[1])
        add_slider_foreach_joint(ref_slider_manager, model, 0, -1.0, 1.0, prefix="b_")
        add_slider_foreach_joint(ref_slider_manager, model, 0, 0, 0.5, prefix="A_")
        add_slider_foreach_joint(ref_slider_manager, model, 0, 0, 2.0, prefix="f_")
        kp_slider_manager.start()
        kd_slider_manager.start()
        ref_slider_manager.start()
        
        
        while viewer.is_running():
            while not SIMULATION_RUNNING:
                time.sleep(0.1)
            step_start = time.time()

            # GET CONTROLLER PARAMETERS
            # kp = np.asarray(kp_slider_manager.get_slider_values(blocking=False))
            # kd = np.asarray(kd_slider_manager.get_slider_values(blocking=False))
            
            # GET REFERENCE
            
            # set planner state
            agent.set_state(
                time=data.time,
                qpos=data.qpos,
                qvel=data.qvel,
                act=data.act,
                mocap_pos=data.mocap_pos,
                mocap_quat=data.mocap_quat,
                userdata=data.userdata,
            )
            if i % steps_per_planning_iteration == 0:
                agent.planner_step()
            # traj = agent.best_trajectory()
            # traj_states = traj['states']
            # traj_actions = traj['actions']
            # q_ref = traj[1,7:26] #19
            # q_dot_ref = traj[1,32:51] #19
            
            
            
            # reference from slider
            # slider_values = ref_slider_manager.get_slider_values(blocking=False)
            # b = np.asarray(slider_values[:19])
            # A = np.asarray(slider_values[19:38])
            # f = np.asarray(slider_values[38:])
            # q_ref = A * np.sin(2*np.pi*data.time*f) + b
            # q_dot_ref = A*2*np.pi*f*np.cos(2*np.pi*data.time*f)
            
            # READ STATE
            q = data.qpos[7:26]
            q_dot = data.qvel[6:25]
            
            
            
            #CONTROL LAW
            
            # set ctrl from agent policy
            data.ctrl = agent.get_action(nominal_action=True)
            
            mujoco.mj_step1(model, data)
            # data.ctrl = np.multiply((q_ref-q), kp) + np.multiply((q_dot_ref-q_dot), kd)
            # data.ctrl += data.qfrc_bias[6:]
            #data.ctrl = traj_actions[0, :]
            plot_manager.send_data(2, (time.time(), data.ctrl))
            # SIMULATION STEP
            
            mujoco.mj_step2(model, data)
            plot_manager.send_data(0, (time.time(), [mujoco.mju_norm(data.body('right_ankle_link').xpos[:2] - data.body('left_ankle_link').xpos[:2]),
                                                     *data.sensor('foot_right_ang_velocity').data,
                                                     data.body('left_ankle_link').xpos[2],
                                                     data.body('right_ankle_link').xpos[2]]))
            #plot_manager.send_data(1, (time.time(), q_ref-data.qpos[7:26]))
            
            # update graphics
            geom_manager.update()
            
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()
            
            # update mocap if goal reached
            if np.linalg.norm(data.sensor('torso_position').data[:2] - data.mocap_pos[0][:2]) < 0.1:
                current_mocap = (current_mocap + 1) % len(mocap_path)
                data.mocap_pos[0] = mocap_path[current_mocap]
            
            i+=1
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
            elif i < 1:
                print("Simulation not running in real time!")

if __name__ == "__main__":
    main()