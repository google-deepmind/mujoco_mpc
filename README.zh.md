<h1>
  <a href="#"><img alt="MuJoCo MPC" src="docs/assets/banner.png" width="100%"></a>
</h1>

<p>
  <a href="https://github.com/google-deepmind/mujoco_mpc/actions/workflows/build.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/google-deepmind/mujoco_mpc/build.yml?branch=main">
  </a>
  <a href="https://github.com/google-deepmind/mujoco_mpc/blob/main/LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/google-deepmind/mujoco_mpc">
  </a>
</p>

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

**MuJoCo MPC (MJPC)** 是一个由 Google DeepMind 开发的交互式应用程序与软件框架，用于基于 [MuJoCo](https://mujoco.org/) 物理引擎的实时预测控制（Real-time Predictive Control）。

MJPC 允许用户轻松设计并求解复杂的机器人控制任务，目前支持多种基于打靶法（shooting-based）的规划器。基于导数的方法包括 iLQG（迭代线性二次高斯）与梯度下降（Gradient Descent），无导数方法则包括一种简洁却极具竞争力的规划器——预测采样（Predictive Sampling）。

- [概览 (Overview)](#概览)
- [图形用户界面 (Graphical User Interface)](#图形用户界面)
- [安装指南 (Installation)](#安装指南)
  - [macOS](#macos)
  - [Ubuntu](#ubuntu-2004)
  - [构建排错 (Build Issues)](#构建排错)
- [Python API](#python-api)
- [预测控制 (Predictive Control)](#预测控制)
- [贡献指南 (Contributing)](#贡献指南)
- [已知问题 (Known Issues)](#已知问题)
- [引用 (Citation)](#引用)
- [致谢 (Acknowledgments)](#致谢)
- [许可证与免责声明 (License and Disclaimer)](#许可证与免责声明)

## 概览

如需阅读介绍此软件包的学术论文，请参阅我们的[预印本论文](https://arxiv.org/abs/2212.00541)。

点击下方观看 MJPC 的快速视频介绍：

[![Video](http://img.youtube.com/vi/Bdx7DuAMB6o/hqdefault.jpg)](https://dpmd.ai/mjpc)

点击下方观看我们在 2022 年 12 月麻省理工学院机器人学术研讨会（MIT Robotics Seminar）上介绍成果的完整讲座：

[![2022Talk](http://img.youtube.com/vi/2xVN-qY78P4/hqdefault.jpg)](https://www.youtube.com/watch?v=2xVN-qY78P4)

2023 年 12 月在 IEEE 基于模型优化技术委员会（IEEE Technical Committee on Model-Based Optimization）上的最新演讲视频可在此观看：

[![2023Talk](https://img.youtube.com/vi/J-JO-lgaKtw/hqdefault.jpg)](https://www.youtube.com/watch?v=J-JO-lgaKtw&t=0s)

### 示例任务

四足机器人任务：

[![Quadruped](http://img.youtube.com/vi/esLuwaWz4oE/hqdefault.jpg)](https://www.youtube.com/watch?v=esLuwaWz4oE)

双臂协同操作：

[![Bimanual](http://img.youtube.com/vi/aCNCKVThKIE/hqdefault.jpg)](https://www.youtube.com/watch?v=aCNCKVThKIE)

魔方 10 步还原：

[![Unscramble](http://img.youtube.com/vi/ZRRvVWV-Muk/hqdefault.jpg)](https://www.youtube.com/watch?v=ZRRvVWV-Muk)

人形机器人动作捕捉追踪：

[![Tracking](http://img.youtube.com/vi/tEBVK-MO1Sw/hqdefault.jpg)](https://www.youtube.com/watch?v=tEBVK-MO1Sw)

## 图形用户界面

有关图形用户界面的详细深入指南，请参阅 [MJPC GUI](docs/GUI.md) 官方文档。

## 安装指南

MJPC 已在 [Ubuntu 20.04](https://releases.ubuntu.com/focal/) 和 [macOS-12](https://www.apple.com/by/macos/monterey/) 上完成完整测试。原则上，其他系统版本与 Windows 操作系统亦可运行 MJPC，但尚未经过官方测试。

### 环境依赖

操作系统专属依赖项：

#### macOS
安装 [Xcode](https://developer.apple.com/xcode/)。

安装 `ninja` 与 `zlib`：
```sh
brew install ninja zlib
```

#### Ubuntu 20.04
```sh
sudo apt-get update && sudo apt-get install cmake libgl1-mesa-dev libxinerama-dev libxcursor-dev libxrandr-dev libxi-dev ninja-build zlib1g-dev clang-12
```

### 克隆 MuJoCo MPC
```sh
git clone https://github.com/google-deepmind/mujoco_mpc
```

### 构建并运行 MJPC GUI 应用程序
1. 切换目录：
```sh
cd mujoco_mpc
```

2. 创建并切换至构建目录：
```sh
mkdir build
cd build
```

3. 配置构建：

#### macOS-12
```sh
cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON
```

#### Ubuntu 20.04
```sh
cmake .. -DCMAKE_BUILD_TYPE:STRING=Release -G Ninja -DCMAKE_C_COMPILER:STRING=clang-12 -DCMAKE_CXX_COMPILER:STRING=clang++-12 -DMJPC_BUILD_GRPC_SERVICE:BOOL=ON
```
**注意：gRPC 是一个较大的依赖库，初次下载可能需要 10-20 分钟。**

4. 执行编译
```sh
cmake --build . --config=Release
```

5. 运行 GUI 应用程序
```sh
cd bin
./mjpc
```

### 使用 VSCode 构建并运行 MJPC GUI 应用程序
我们推荐使用 [VSCode](https://code.visualstudio.com/) 及其两款官方插件（[CMake Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cmake-tools) 与 [C/C++](https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools)）来简化构建流程：

1. 在 VSCode 中打开克隆的目录 `mujoco_mpc`。
2. 使用 CMake 配置项目（VSCode 会自动弹出提示）。
3. 将编译器配置为 `clang-12`。
4. 以 Release 模式构建并运行 `mjpc` 目标（VSCode 默认通常为 Debug 模式）。这将自动编译并启动图形用户界面。

### 构建排错
如果遇到构建问题，请查阅 [GitHub Actions 持续集成配置文件](https://github.com/google-deepmind/mujoco_mpc/blob/main/.github/workflows/build.yml)。该文件展示了我们在 Ubuntu 20.04 和 macOS-12 上构建与测试 MJPC 所使用的完整确切环境配置。

# Python API

我们为 MJPC 提供了简易的 Python API。该 API 目前仍处于实验阶段，要求使用者具备一定的使用经验。例如，正确使用要求在 Python 中定义的模型与 MJPC 任务（即在 C++ 中定义的残差函数和状态转移函数）彼此兼容。目前，Python API 并未提供专门用于验证这种兼容性的错误处理机制，若没有对 MuJoCo 和 MJPC 的深入了解，可能会较难进行排错。

## 安装指南

### 环境准备
1. 构建 MJPC（详见上述构建说明）。

2. Python 3.10

3. （可选）创建包含 **Python 3.10** 的 Conda 环境：
```sh
conda create -n mjpc python=3.10
conda activate mjpc
```

4. 安装 MuJoCo：
```sh
pip install mujoco
```

### 安装 API 模块
接下来，切换到 `python` 目录：
```sh
cd python
```

安装 Python 模块：
```sh
python setup.py install
```

验证安装是否成功：
```sh
python "mujoco_mpc/agent_test.py"
```

示例脚本位于 `python/mujoco_mpc/demos` 目录。例如在 `python/` 目录下运行：
```sh
python mujoco_mpc/demos/agent/cartpole_gui.py
```
将通过 Python 调用 MuJoCo 的被动查看器（passive viewer）启动 MJPC GUI 应用程序。

### Python API 安装排错
如果安装失败或提前终止，我们建议直接删除 MJPC `build` 构建目录并从头重新构建，因为构建文件很可能已被损坏。此外，请从 `python/` 目录中清理在安装过程中生成的临时文件。

## 预测控制

更多详情请参阅[预测控制 (Predictive Control)](docs/OVERVIEW.md) 官方文档。

## 贡献指南

更多详情请参阅[贡献指南 (Contributing)](docs/CONTRIBUTING.md) 官方文档。

## 已知问题

MJPC 并非生产级工业软件，它是一个**科研原型（research prototype）**。代码中可能存在缺失的功能特性或潜在的 bug。如果您发现任何问题，请在[问题追踪器](https://github.com/google-deepmind/mujoco_mpc/issues)中提交 Issue。以下列出部分已知问题（包括我们正在积极推进改进的方面）：

- 我们尚未在 Windows 系统上对 MJPC 进行完整测试，但原则上应不存在阻碍运行的问题。
- 任务规范定义较为繁琐，尤其是在 XML 中设置范数（norms）及其参数的方式略显笨重。我们仍在持续迭代相关设计。
- 梯度下降的搜索步长与代价函数（cost function）的量纲尺度成正比，需要针对每个具体任务进行调优方能获得最佳效果。这并非代码 bug，而是标准梯度下降法的固有特性。通过某种梯度归一化机制可能有助于缓解此问题，但我们目前尚未进行全面深入的探究。

## 引用

如果您的研究工作使用了 MJPC，请引用我们关联的[预印本论文](https://arxiv.org/abs/2212.00541)：

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

## 致谢

使本开源仓库顺利公开的主要工作由 [Taylor Howell](https://thowell.github.io/) 及 Google DeepMind 机器人仿真团队（Robotics Simulation team）承担与完成。

## 许可证与免责声明

所有其他内容版权归 DeepMind Technologies Limited 2022 所有，并在 Apache License 2.0 许可证下开源。许可证副本可在本仓库根目录下的 LICENSE 文件中查看，亦可在 https://www.apache.org/licenses/LICENSE-2.0 获取。

本仓库不是 Google 的官方正式支持产品（This is not an officially supported Google product）。

---

> **翻译说明（Translation Disclaimer）**
> 本文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）人工翻译并维护，旨在降低中文开发者的阅读门槛。最后同步日期：2026年10月23日。若译文与英文原版存在任何出入或滞后，请以 [英文原版 (README.md)](README.md) 为准。欢迎通过 PR 协助改进译文！
