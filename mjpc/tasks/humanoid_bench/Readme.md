# HumanoidBench in MuJoCo-MPC

This directory contains the re-implementation of some of the tasks of the HumanoidBench benchmark in MuJoCo-MPC. The original implementation is in [this repository](https://github.com/carlosferrazza/humanoid-bench).

## Reward to Residuals
MuJoCo-MPC uses residuals with multiple dimensions instead of a single reward.
The residuals should be 'close to zero' to indicate a good performance. So in each task, the first step is to compute the reward the same way it is done in the original implementation. 
Then, the first dimension of the residual is set to x - reward, where x is the maximum reward that can be achieved in the task.

## Additional Residuals
In addition to the reward residual, we also add additional residuals. We found them to be helpful to solve the task. 
To get the 'vanilla' version of the task, you can set the additional residuals weights to zero, using the sliders in the GUI.

## Robots
In the original implementation, they use a position controlled H1 robot from unitree.
In addition, in some of the tasks we also include the G1 robot from unitree. This robot is torque controlled.
