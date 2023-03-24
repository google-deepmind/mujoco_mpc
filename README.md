<h1>
  <a href="#"><img alt="MuJoCo MPC" src="docs/assets/banner.png" width="100%"></a>
</h1>

<p>
  <a href="https://github.com/deepmind/mujoco_mpc/actions/workflows/build.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/deepmind/mujoco_mpc/build.yml?branch=main">
  </a>
  <a href="https://github.com/deepmind/mujoco_mpc/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/deepmind/mujoco_mpc">
  </a>
</p>

**MuJoCo MPC (MJPC)** is an interactive application and software framework for real-time predictive control with [MuJoCo](https://mujoco.org/), developed by DeepMind.

MJPC allows the user to easily author and solve complex robotics tasks, and currently supports three shooting-based planners: derivative-based iLQG and Gradient Descent, and a simple yet very competitive derivative-free method called Predictive Sampling.

- [Overview](#overview)
- [Graphical User Interface](#graphical-user-interface)
- [Installation](#installation)
  - [macOS](#macos)
  - [Ubuntu](#ubuntu)
  - [Build Issues](#build-issues)
- [Predictive Control](#predictive-control)
- [Contributing](#contributing)
- [Known Issues](#known-issues)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [License and Disclaimer](#license-and-disclaimer)

## Overview

To read the paper describing this software package, please see out [preprint](https://arxiv.org/abs/2212.00541).

For a quick video overview of MJPC, click below.

[![Video](http://img.youtube.com/vi/Bdx7DuAMB6o/hqdefault.jpg)](https://dpmd.ai/mjpc)

For a longer talk at the MIT Robotics Seminar describing our results, click below.

[![Talk](http://img.youtube.com/vi/2xVN-qY78P4/hqdefault.jpg)](https://www.youtube.com/watch?v=2xVN-qY78P4)

## Graphical User Interface

For a detailed dive of the graphical user interface, see the [MJPC GUI](docs/GUI.md) documentation.

## Installation

You will need [CMake](https://cmake.org/) and a working C++20 compiler to build MJPC. We recommend using [VSCode](https://code.visualstudio.com/) and 2 of its extensions ([CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) and [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)) to simplify the build process.

1. Clone the repository: `git clone https://github.com/deepmind/mujoco_mpc.git`
2. Configure the project with CMake (a pop-up should appear in VSCode)
3. Build and run the `mjpc` target in "release" mode (VSCode defaults to "debug"). This will open and run the graphical user interface.

### macOS
Additionally, install [Xcode](https://developer.apple.com/xcode/).

### Ubuntu
Additionally, install:
```shell
sudo apt-get install libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build
```

### Build Issues
If you encounter build issues, please see the
[Github Actions configuration](https://github.com/deepmind/mujoco_mpc/blob/main/.github/workflows/build.yml).
Note, we are using `clang-14`.

## Predictive Control

See the [Predictive Control](docs/OVERVIEW.md) documentation for more information.

## Contributing

See the [Contributing](docs/CONTRIBUTING.md) documentation for more information.

## Known Issues

MJPC is not production-quality software, it is a **research prototype**. There are likely to be missing features and outright bugs. If you find any, please report them in the [issue tracker](https://github.com/deepmind/mujoco_mpc/issues). Below we list some known issues, including items that we are actively working on.

- We have not tested MJPC on Windows, but there should be no issues in principle.
- Task specification, in particular the setting of norms and their parameters in XML, is a bit clunky. We are still iterating on the design.
- The Gradient Descent search step is proportional to the scale of the cost function and requires per-task tuning in order to work well. This is not a bug but a property of vanilla gradient descent. It might be possible to ameliorate this with some sort of gradient normalisation, but we have not investigated this thoroughly.

## Citation

If you use MJPC in your work, please cite our accompanying [preprint](https://arxiv.org/abs/2212.00541):

```bibtex
@article{howell2022,
  title={{Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo}},
  author={Howell, Taylor and Gileadi, Nimrod and Tunyasuvunakool, Saran and Zakka, Kevin and Erez, Tom and Tassa, Yuval},
  archivePrefix={arXiv},
  eprint={2212.00541},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2212.00541},
  doi={10.48550/arXiv.2212.00541},
  year={2022},
  month={dec}
}
```

## Acknowledgments

The main effort required to make this repository publicly available was undertaken by [Taylor Howell](https://thowell.github.io/) and the DeepMind Robotics Simulation team.

## License and Disclaimer

All other content is Copyright 2022 DeepMind Technologies Limited and licensed under the Apache License, Version 2.0. A copy of this license is provided in the top-level LICENSE file in this repository. You can also obtain it from https://www.apache.org/licenses/LICENSE-2.0.

This is not an officially supported Google product.
