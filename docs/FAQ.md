# Frequently Asked Questions

Here we present some information that may not be immediately obvious for new users. 

## GUI

- Press `F1` to bring up a help pane that describes how to use the GUI.
- The MJPC GUI is an extension of MuJoCo's native `simulate` viewer, with the same keyboard shortcuts and mouse functionality. 
- The `simulate` viewer enables drag-and-drop interaction with simulated objects to apply forces or torques. 
    - Double-click on a body to select it. 
    - `Ctrl + left drag` applies a torque to the selected object, resulting in rotation. 
    - `Ctrl + right drag` applies a force to the selected object in the `(x,z)` plane, resulting in translation. 
    - `Ctrl + Shift + right drag` applies a force to the selected object in the `(x,y)` plane. 
- MJPC adds three keyboard shortcuts:
    - The `Enter` key starts and stops the planner.
    - The `\` key starts and stops the controller (sending actions from the planner to the model).
    - The `9` key turns the traces on/off.


