"""Merge the robotiq and panda models."""

import sys


merge_filename = sys.argv[1]
panda_filename = merge_filename.replace('panda_robotiq.xml',
                                        'panda_nohand.xml')
robotiq_filename = merge_filename.replace('panda_robotiq.xml',
                                          '2f85.xml')
with open(panda_filename) as panda_file:
  panda = panda_file.read()
with open(robotiq_filename) as robotiq_file:
  robotiq = robotiq_file.read()

# insert defaults
default_begin_index = robotiq.index('<default>')  # include default tag
last_default_index = robotiq.rindex('</default>')
defaults = robotiq[default_begin_index: last_default_index]
panda = panda.replace('<default>', defaults)

# insert assets
asset_begin_index = robotiq.index('<asset>')  # include asset tag
asset_close_index = robotiq.index('</asset>', asset_begin_index)
assets = robotiq[asset_begin_index:asset_close_index]
panda = panda.replace('<asset>', assets)

# attach model
worldbody_index = robotiq.index('<worldbody>') + len('<worldbody>')
close_worldbody_index = robotiq.index('</worldbody>', worldbody_index)
robotiq_body = robotiq[worldbody_index:close_worldbody_index]
panda = panda.replace('<site name="attachment_site"/>', robotiq_body)

# insert bottom: contact, tendon, equality
contact_begin_index = robotiq.index('</worldbody>')  # include closing tag
equality_close_index = robotiq.index(
    '</equality>', contact_begin_index) + len('</equality>')
bottom = robotiq[contact_begin_index:equality_close_index]
panda = panda.replace('</worldbody>', bottom)

# add gravity compensation to all bodies
panda = panda.replace('<body ', '<body gravcomp="1" ')

# eliminate contact with the target
panda = panda.replace('priority="1"',
                      'priority="1" contype="6" conaffinity="5"')
panda = panda.replace(
    '<geom type="mesh" group="3"/>',
    '<geom type="mesh" group="3" contype="2" conaffinity="1"/>')

# add cartesian actuators
cartesian_actuators = '''
  <actuator>
    <general  name="x" site="pinch" refsite="pedestal" ctrlrange="-.5 .5" ctrllimited="true" gainprm="1000" biasprm="0 -1000 -200" biastype="affine" gear="1 0 0 0 0 0"/>
    <general  name="y" site="pinch" refsite="pedestal" ctrlrange="-.5 .5" ctrllimited="true" gainprm="1000" biasprm="0 -1000 -200" biastype="affine" gear="0 1 0 0 0 0"/>
    <general  name="z" site="pinch" refsite="pedestal" ctrlrange="-.5 .5" ctrllimited="true" gainprm="1000" biasprm="300 -1000 -200" biastype="affine" gear="0 0 1 0 0 0"/>
    <general name="rx" site="pinch" refsite="world"    ctrlrange="-.5 .5" ctrllimited="true" gainprm="100" biasprm="0 -100 -20" biastype="affine"  gear="0 0 0 1 0 0"/>
    <general name="ry" site="pinch" refsite="world"    ctrlrange="-.5 .5" ctrllimited="true" gainprm="100" biasprm="0 -100 -20" biastype="affine"  gear="0 0 0 0 1 0"/>
    <general name="rz" site="pinch" refsite="world"    ctrlrange="-1.5 1.5" ctrllimited="true" gainprm="10" biasprm="0 -10 -2" biastype="affine"  gear="0 0 0 0 0 1"/>
    <position name="fingers" ctrllimited="true" forcelimited="true" ctrlrange="0 1" forcerange="-5 5" kp="40" tendon="split"/>
  </actuator>
'''
actuator_begin_index = panda.index('<actuator>')
actuator_close_index = panda.index(
    '</actuator>', actuator_begin_index) + len('</actuator>')
actuators = panda[actuator_begin_index:actuator_close_index]
panda = panda.replace(actuators, cartesian_actuators)

# remove panda keyframe
keyframe_begin_index = panda.index('<keyframe>')  # keep tag (for removal)
keyframe_close_index = panda.index('</keyframe>') + len('</keyframe>')
panda = panda.replace(panda[keyframe_begin_index:keyframe_close_index], '')

with open(merge_filename, 'w') as merged_file:
  merged_file.write(panda)
