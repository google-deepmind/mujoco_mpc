DESC ="""
Script to randomly generate a kitchen scene
USAGE:
    python random_kitchen.py -m kitchen3.xml
"""
import os
import click
import numpy as np
import time
import re

try:
    import mujoco_py
    from mujoco_py import load_model_from_path, MjSim, MjViewer, load_model_from_xml, functions
except ImportError as e:
    raise ImportError("(HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)")

from robohive.utils.quat_math import euler2quat


def get_sim(model_path:str=None, model_xmlstr=None):
        """
        Get sim using model_path or model_xmlstr.
        """
        if model_path:
            if model_path.startswith("/"):
                fullpath = model_path
            else:
                fullpath = os.path.join(os.path.dirname(__file__), model_path)
            if not os.path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
            model = load_model_from_path(fullpath)
        elif model_xmlstr:
            model = load_model_from_xml(model_xmlstr)
        else:
            raise TypeError("Both model_path and model_xmlstr can't be None")

        return MjSim(model)

# Randomize appliances
def randomize_appliances(model_path, np_random):

    model_file = open(model_path, "r")
    model_xml = model_file.read()
    opt = np_random.randint(low=0, high=4)
    model_xml = re.sub('microwave_body\d.xml','microwave_body{}.xml'.format(opt), model_xml)
    opt = np_random.randint(low=0, high=8)
    model_xml = re.sub('kettle_body\d.xml','kettle_body{}.xml'.format(opt), model_xml)

    # Save new random xml
    processed_model_path = model_path[:-4]+"_random.xml"
    with open(processed_model_path, 'w') as file:
        file.write(model_xml)
    return processed_model_path

def randomize_textures(model_path, np_random):
    # Recover parsed xml
    sim = get_sim(model_path=model_path)
    raw_xml = sim.model.get_xml()

    # Process to get a random xml
    n_tex = 5
    processed_xml = raw_xml
    opt = np_random.randint(low=0, high=n_tex)
    processed_xml = re.sub('wood\d.png','wood{}.png'.format(opt), processed_xml)
    processed_xml = re.sub('stone\d.png','stone{}.png'.format(opt), processed_xml)
    processed_xml = re.sub('metal\d.png','metal{}.png'.format(opt), processed_xml)

    # Save new random xml
    processed_model_path = model_path[:-4]+"_random.xml"
    with open(processed_model_path, 'w') as file:
        file.write(processed_xml)

    sim = get_sim(model_path=processed_model_path)
    return sim


# Randomize layout
def randomize_layout(sim, np_random):

    body_offset = {
        'slidecabinet': {'pos':[0,0,0], 'euler':[0,0,0]},
        'hingecabinet': {'pos':[0,0,0], 'euler':[0,0,0]},
        'microwave': {'pos':[0,0,-0.2], 'euler':[0,0,0]},
        'kettle': {'pos':[0,0,0.0], 'euler':[0,0,0]},
    }

    # Counter layout
    layout = {
            'sink': {
                'L': [-1.620, 0, 0],
                'R': [0, 0, 0]},
            'island': {
                'L': [-.020, 0, 0],
                'R': [1.92, 0, 0]},
    }
    counter_loc = ['L', 'R']

    # Appliance layout
    app_xy = {
        # Front pannel
        'FL':[-.5, 0.28],
        'FR':[.4, 0.28],
        # Left pannel
        'LL':[-1., -1.0],
        'LR':[-1., -.25],
        # Right pannel
        'RL':[1., -.25],
        'RR':[1., -1.0],
    }
    app_z = {
            'T':[2.6],
            'M':[2.2],
            'B':[1.8],
        }
    app_xyz = {
        # Front pannel
        'FLT': {'pos':app_xy['FL']+app_z['T'], 'euler':[0,0,0], 'accept':True},
        'FRT': {'pos':app_xy['FR']+app_z['T'], 'euler':[0,0,0], 'accept':True},
        'FLM': {'pos':app_xy['FL']+app_z['M'], 'euler':[0,0,0], 'accept':False},
        'FRM': {'pos':app_xy['FR']+app_z['M'], 'euler':[0,0,0], 'accept':False},
        'FLB': {'pos':app_xy['FL']+app_z['B'], 'euler':[0,0,0], 'accept':False},
        'FRB': {'pos':app_xy['FR']+app_z['B'], 'euler':[0,0,0], 'accept':False},
        # Left pannel
        'LLT': {'pos':app_xy['LL']+app_z['T'], 'euler':[0,0,1.57], 'accept':True},
        'LRT': {'pos':app_xy['LR']+app_z['T'], 'euler':[0,0,1.57], 'accept':True},
        'LLM': {'pos':app_xy['LL']+app_z['M'], 'euler':[0,0,1.57], 'accept':True},
        'LRM': {'pos':app_xy['LR']+app_z['M'], 'euler':[0,0,1.57], 'accept':False},
        'LLB': {'pos':app_xy['LL']+app_z['B'], 'euler':[0,0,1.57], 'accept':True},
        'LRB': {'pos':app_xy['LR']+app_z['B'], 'euler':[0,0,1.57], 'accept':True},
        # Right pannel
        'RLT': {'pos':app_xy['RL']+app_z['T'], 'euler':[0,0,-1.57], 'accept':True},
        'RRT': {'pos':app_xy['RR']+app_z['T'], 'euler':[0,0,-1.57], 'accept':True},
        'RLM': {'pos':app_xy['RL']+app_z['M'], 'euler':[0,0,-1.57], 'accept':False},
        'RRM': {'pos':app_xy['RR']+app_z['M'], 'euler':[0,0,-1.57], 'accept':True},
        'RLB': {'pos':app_xy['RL']+app_z['B'], 'euler':[0,0,-1.57], 'accept':True},
        'RRB': {'pos':app_xy['RR']+app_z['B'], 'euler':[0,0,-1.57], 'accept':True},
    }
    app_loc = [*app_xyz] # list of dict keys

    # Randomize counter layouts
    opt = np_random.randint(low=0, high=2)
    sel_grid = counter_loc[opt]
    # Place island
    bid = sim.model.body_name2id('island')
    sim.model.body_pos[bid] = layout['island'][sel_grid]
    # place sink
    sel_grid = counter_loc[1-opt]
    bid = sim.model.body_name2id('sink')
    sim.model.body_pos[bid] = layout['sink'][sel_grid]
    # Don't mount anything next to sink
    for side in ['L','R']:
        app_xyz[sel_grid+side+'B']['accept'] = False

    # Randomize Appliances
    for body_name in ['slidecabinet', 'hingecabinet', 'microwave']:
        # Find and empty slot
        empty_slot = False
        while not empty_slot:
            opt = np_random.randint(low=0, high=len(app_loc))
            sel_grid = app_loc[opt]
            empty_slot = True if app_xyz[sel_grid]['accept'] else False
        bid = sim.model.body_name2id(body_name)
        sim.model.body_pos[bid] = np.array(app_xyz[sel_grid]['pos'])+np.array(body_offset[body_name]['pos'])
        sim.model.body_quat[bid] = euler2quat(np.array(app_xyz[sel_grid]['euler']) + np.array(body_offset[body_name]['euler']))
        # mark occupied
        app_xyz[sel_grid]['accept'] = False
        # handle corner assignments
        if sel_grid in ['LRT', 'FLT']:
            app_xyz['LRT']['accept'] = app_xyz['FLT']['accept'] = False
        if sel_grid in ['FRT', 'RLT']:
            app_xyz['FRT']['accept'] = app_xyz['RLT']['accept'] = False


# Randomize visuals
def randomize_visuals(sim, np_random):
    # randomize colors
    sim.model.geom_rgba[:,:] =np_random.uniform(size=sim.model.geom_rgba.shape, low=0.4, high=.8)
    sim.model.geom_rgba[:,3] =1

    # randomize lights/ surface properties etc [ToDO]


@click.command(help=DESC)
@click.option('-m', '--model_path', type=str, help='full path of environment xml to load', required= True)
@click.option('-s', '--seed', type=int, help='seed for randomizaton', default=0)
def randomize_scene(model_path, seed):

    np_random = np.random
    np_random.seed(seed)

    while True:
        processed_model_path = randomize_appliances(model_path, np_random)
        sim = randomize_textures(processed_model_path, np_random)
        viewer = MjViewer(sim)
        for _ in range(1000):
            randomize_layout(sim, np_random)
            # randomize_visuals(sim, np_random)
            sim.forward()
            viewer.render()

if __name__ == '__main__':
    randomize_scene()