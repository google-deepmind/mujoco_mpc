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
# ==============================================================================
"""Install script for MuJoCo MPC."""

import pathlib
import shutil

import grpc_tools.protoc
import pkg_resources
import setuptools
from setuptools.command.develop import develop
from setuptools.command.install import install


class CompileProtoGrpcCommand(setuptools.Command):
  """Specialized setup command to handle agent proto compilation.

  Compiles the `agent_pb2{_grpc}.py` files from `agent_proto`. Assumes that
  `grpc_tools.protoc` is installed.
  """

  description = "Compile `.proto` files to Python protobuf and gRPC files."
  user_options = []

  def initialize_options(self):
    ...

  def finalize_options(self):
    ...

  def run(self):
    """Compile `agent.proto` into `agent_pb2{_grpc}.py`.

    This function looks more complicated than what it has to be because the
    `protoc` compiler is very particular in the way it generates the imports for
    the generated `agent_pb2_grpc.py` file. The final argument of the `protoc`
    call has to be "mujoco_mpc/agent.proto" in order for the import to become
    `from mujoco_mpc import [agent_pb2_proto_import]` instead of just
    `import [agent_pb2_proto_import]`. The latter would fail because the name is
    meant to be relative but python3 interprets it as an absolute import.
    """
    agent_proto_filename = "agent.proto"
    agent_proto_source_path = pathlib.Path(
        "../grpc", agent_proto_filename
    ).resolve()
    agent_proto_destination_path = pathlib.Path(
        pkg_resources.resource_filename("mujoco_mpc", agent_proto_filename)
    ).resolve()
    # Copy `agent_proto_filename` into current source.
    shutil.copy(agent_proto_source_path, agent_proto_destination_path)

    # Compile with `protoc`, explicitly defining a relative target to get the
    # import correct.
    grpc_tools.protoc.main([
        f"-I{agent_proto_destination_path.parent.parent}",
        f"--python_out={agent_proto_destination_path.parent.parent}",
        f"--grpc_python_out={agent_proto_destination_path.parent.parent}",
        f"mujoco_mpc/{agent_proto_filename}",
    ])


class CopyAgentServiceBinaryCommand(setuptools.Command):
  """Specialized setup command to copy `agent_service` next to `agent.py`.

  Assumes that the C++ gRPC `agent_service` binary has been manually built and
  and located in the default `mujoco_mpc/build/bin` folder.
  """

  description = "Copy `agent_service` next to `agent.py`."
  user_options = []

  def initialize_options(self):
    ...

  def finalize_options(self):
    ...

  def run(self):
    source_path = pathlib.Path("../build/bin/agent_service")
    if not source_path.exists():
      raise ValueError(
          f"Cannot find `agent_service` binary from {source_path}. Please build"
          " the `agent_service` C++ gRPC service."
      )
    destination_path = pathlib.Path(
        pkg_resources.resource_filename("mujoco_mpc", "mjpc/agent_service")
    )

    self.announce(f"{source_path.resolve()=}")
    self.announce(f"{destination_path.resolve()=}")

    destination_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(source_path, destination_path)


class CopyTaskAssetsCommand(setuptools.Command):
  """Specialized setup command to copy `agent_service` next to `agent.py`.

  Assumes that the C++ gRPC `agent_service` binary has been manually built and
  and located in the default `mujoco_mpc/build/bin` folder.
  """

  description = (
      "Copy task assets over to python source to make them accessible by"
      " `Agent`."
  )
  user_options = []

  def initialize_options(self):
    ...

  def finalize_options(self):
    ...

  def run(self):
    mjpc_tasks_path = pathlib.Path(__file__).parent.parent / "mjpc" / "tasks"
    source_paths = tuple(mjpc_tasks_path.rglob("*.xml"))
    relative_source_paths = tuple(
        p.relative_to(mjpc_tasks_path) for p in source_paths
    )
    destination_dir_path = pathlib.Path(
        pkg_resources.resource_filename("mujoco_mpc", "mjpc/tasks")
    )
    print(f"{mjpc_tasks_path=}")
    print(f"{relative_source_paths=}")
    print(f"{destination_dir_path=}")
    self.announce(
        f"Copying assets {relative_source_paths} from"
        f" {mjpc_tasks_path} over to {destination_dir_path}"
    )

    for source_path, relative_source_path in zip(
        source_paths, relative_source_paths
    ):
      destination_path = destination_dir_path / relative_source_path
      destination_path.parent.mkdir(exist_ok=True, parents=True)
      shutil.copy(source_path, destination_path)


class InstallCommand(install):
  """Specialized Python builder to handle agent service dependencies.

  During installation, this will comile the `agent_pb2{_grpc}.py` files and copy
  `agent_service` binary next to `agent.py`.
  """

  user_options = install.user_options

  def run(self):
    self.run_command("compile_proto_grpc")
    self.run_command("copy_agent_service_binary")
    self.run_command("copy_task_assets")
    install.run(self)


class DevelopCommand(develop):
  """Specialized Python builder to handle agent service dependencies.

  During installation, this will comile the `agent_pb2{_grpc,}.py` files and
  copy `agent_service` binary next to `agent.py`.
  """

  user_options = develop.user_options

  def run(self):
    self.run_command("compile_proto_grpc")
    self.run_command("copy_agent_service_binary")
    install.run(self)


setuptools.setup(
    name="mujoco_mpc",
    version="0.1.0",
    author="DeepMind",
    author_email="mujoco@deepmind.com",
    description="MuJoCo MPC (MJPC)",
    url="https://github.com/deepmind/mujoco_mpc",
    license="MIT",
    classifiers=[
        # TODO(khartikainen): Check these
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Topic :: Scientific/Engineering",
    ],
    packages=setuptools.find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "grpcio-tools >= 1.53.0",
        "grpcio >= 1.53.0",
    ],
    extras_require={
        "test": [
            "absl-py",
            "mujoco >= 2.3.3",
        ],
    },
    cmdclass={
        "develop": DevelopCommand,
        "install": InstallCommand,
        "compile_proto_grpc": CompileProtoGrpcCommand,
        "copy_agent_service_binary": CopyAgentServiceBinaryCommand,
        "copy_task_assets": CopyTaskAssetsCommand,
    },
    package_data={
        "": ["mjpc/agent_service", "mjpc/tasks/**/*.xml"],
    },
)
