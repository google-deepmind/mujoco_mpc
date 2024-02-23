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

import os
import pathlib
import platform
import shutil
import setuptools
from setuptools.command import build_ext
from setuptools.command import build_py
import subprocess


Path = pathlib.Path


class GenerateProtoGrpcCommand(setuptools.Command):
  """Specialized setup command to handle direct proto compilation.

  Generates the `direct_pb2{_grpc}.py` files from `direct_proto`. Assumes that
  `grpc_tools.protoc` is installed.
  """

  description = "Generate `.proto` files to Python protobuf and gRPC files."
  user_options = []

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options("build_py", ("build_lib", "build_lib"))

  def run(self):
    """Generate `direct.proto` into `direct_pb2{_grpc}.py`.

    This function looks more complicated than what it has to be because the
    `protoc` generator is very particular in the way it generates the imports
    for the generated `direct_pb2_grpc.py` file. The final argument of the
    `protoc` call has to be "mujoco_mpc/direct.proto" in order for the import to
    become `from mujoco_mpc import [direct_pb2_proto_import]` instead of just
    `import [direct_pb2_proto_import]`. The latter would fail because the name is
    meant to be relative but python3 interprets it as an absolute import.
    """
    # We import here because, if the import is at the top of this file, we
    # cannot resolve the dependencies without having `grpcio-tools` installed.
    from grpc_tools import protoc  # pylint: disable=import-outside-toplevel

    direct_proto_filename = "direct.proto"
    direct_proto_source_path = Path(
        "..", "mjpc", "grpc", direct_proto_filename
    ).resolve()
    assert self.build_lib is not None
    build_lib_path = Path(self.build_lib).resolve()
    proto_module_relative_path = Path(
        "mujoco_mpc", "proto", direct_proto_filename
    )
    direct_proto_destination_path = Path(
        build_lib_path, proto_module_relative_path
    )
    direct_proto_destination_path.parent.mkdir(parents=True, exist_ok=True)
    # Copy `direct_proto_filename` into current source.
    shutil.copy(direct_proto_source_path, direct_proto_destination_path)

    protoc_command_parts = [
        # We use `__file__`  as the first argument the same way as is done by
        # `protoc` when called as `__main__` here:
        # https://github.com/grpc/grpc/blob/21996c37842035661323c71b9e7040345f0915e2/tools/distrib/python/grpcio_tools/grpc_tools/protoc.py#L172-L173.
        __file__,
        f"-I{build_lib_path}",
        f"--python_out={build_lib_path}",
        f"--grpc_python_out={build_lib_path}",
        str(direct_proto_destination_path),
    ]

    protoc_returncode = protoc.main(protoc_command_parts)

    if protoc_returncode != 0:
      raise subprocess.CalledProcessError(
          returncode=protoc_returncode,
          cmd=f"`protoc.main({protoc_command_parts})`",
      )

    self.spawn([
        "touch",
        str(direct_proto_destination_path.parent / "__init__.py"),
    ])


class CopyDirectServerBinaryCommand(setuptools.Command):
  """Specialized setup command to copy `direct_server` next to `direct.py`.

  Assumes that the C++ gRPC `direct_server` binary has been manually built and
  and located in the default `mujoco_mpc/build/bin` folder.
  """

  description = "Copy `direct_server` next to `direct.py`."
  user_options = []

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options("build_py", ("build_lib", "build_lib"))

  def run(self):
    self._copy_binary("direct_server")
    # self._copy_binary("ui_direct_server")

  def _copy_binary(self, binary_name):
    source_path = Path(f"../build/bin/{binary_name}")
    if not source_path.exists():
      raise ValueError(
          f"Cannot find `{binary_name}` binary from {source_path}. Please build"
          " the `{binary_name}` C++ gRPC service."
      )
    assert self.build_lib is not None
    build_lib_path = Path(self.build_lib).resolve()
    destination_path = Path(build_lib_path, "mujoco_mpc", "mjpc", binary_name)

    self.announce(f"{source_path.resolve()=}")
    self.announce(f"{destination_path.resolve()=}")

    destination_path.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy(source_path, destination_path)


class CopyTaskAssetsCommand(setuptools.Command):
  """Copies `direct_server` and `ui_direct_server` next to `direct.py`.

  Assumes that the C++ gRPC `direct_server` binary has been manually built and
  and located in the default `mujoco_mpc/build/bin` folder.
  """

  description = (
      "Copy task assets over to python source to make them accessible by"
      " `Direct`."
  )
  user_options = []

  def initialize_options(self):
    self.build_lib = None

  def finalize_options(self):
    self.set_undefined_options("build_ext", ("build_lib", "build_lib"))

  def run(self):
    mjpc_tasks_path = Path(__file__).parent.parent / "mjpc" / "tasks"
    source_paths = tuple(mjpc_tasks_path.rglob("*.xml"))
    relative_source_paths = tuple(
        p.relative_to(mjpc_tasks_path) for p in source_paths
    )
    assert self.build_lib is not None
    build_lib_path = Path(self.build_lib).resolve()
    destination_dir_path = Path(build_lib_path, "mujoco_mpc", "mjpc", "tasks")
    self.announce(
        f"Copying assets {relative_source_paths} from"
        f" {mjpc_tasks_path} over to {destination_dir_path}."
    )

    for source_path, relative_source_path in zip(
        source_paths, relative_source_paths
    ):
      destination_path = destination_dir_path / relative_source_path
      destination_path.parent.mkdir(exist_ok=True, parents=True)
      shutil.copy(source_path, destination_path)


class BuildPyCommand(build_py.build_py):
  """Specialized Python builder to handle direct service dependencies.

  During build, this will generate the `direct_pb2{_grpc}.py` files and copy
  `direct_server` binary next to `direct.py`.
  """

  user_options = build_py.build_py.user_options

  def run(self):
    self.run_command("generate_proto_grpc")
    self.run_command("copy_task_assets")
    super().run()


class CMakeExtension(setuptools.Extension):
  """A Python extension that has been prebuilt by CMake.

  We do not want distutils to handle the build process for our extensions, so
  so we pass an empty list to the super constructor.
  """

  def __init__(self, name):
    super().__init__(name, sources=[])


class BuildCMakeExtension(build_ext.build_ext):
  """Uses CMake to build extensions."""

  def run(self):
    self._configure_and_build_direct_server()
    self.run_command("copy_direct_server_binary")

  def _configure_and_build_direct_server(self):
    """Check for CMake."""
    cmake_command = "cmake"
    build_cfg = "Debug"
    mujoco_mpc_root = Path(__file__).parent.parent
    mujoco_mpc_build_dir = mujoco_mpc_root / "build"
    cmake_configure_args = [
        "-DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE",
        f"-DCMAKE_BUILD_TYPE:STRING={build_cfg}",
        "-DBUILD_TESTING:BOOL=OFF",
        "-DMJPC_BUILD_GRPC_SERVICE:BOOL=ON",
    ]

    if platform.system() == "Darwin" and "ARCHFLAGS" in os.environ:
      osx_archs = []
      if "-arch x86_64" in os.environ["ARCHFLAGS"]:
        osx_archs.append("x86_64")
      if "-arch arm64" in os.environ["ARCHFLAGS"]:
        osx_archs.append("arm64")
      cmake_configure_args.append(
          f"-DCMAKE_OSX_ARCHITECTURES={';'.join(osx_archs)}"
      )

    # TODO(hartikainen): We currently configure the builds into
    # `mujoco_mpc/build`. This should use `self.build_{temp,lib}` instead, to
    # isolate the Python builds from the C++ builds.
    print("Configuring CMake with the following arguments:")
    for arg in cmake_configure_args:
      print(f"  {arg}")
    subprocess.check_call(
        [
            cmake_command,
            *cmake_configure_args,
            f"-S{mujoco_mpc_root.resolve()}",
            f"-B{mujoco_mpc_build_dir.resolve()}",
        ],
        cwd=mujoco_mpc_root,
    )

    print("Building `direct_server` and `ui_direct_server` with CMake")
    subprocess.check_call(
        [
            cmake_command,
            "--build",
            str(mujoco_mpc_build_dir.resolve()),
            "--target",
            "direct_server",
            # "ui_direct_server",
            f"-j{os.cpu_count()}",
            "--config",
            build_cfg,
        ],
        cwd=mujoco_mpc_root,
    )


setuptools.setup(
    name="mujoco_mpc",
    version="0.1.0",
    author="DeepMind",
    author_email="mujoco@deepmind.com",
    description="MuJoCo MPC (MJPC)",
    url="https://github.com/google-deepmind/mujoco_mpc",
    license="MIT",
    classifiers=[
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
        "grpcio-tools",
        "grpcio",
    ],
    extras_require={
        "test": [
            "absl-py",
            "mujoco >= 3.1.1",
            "mujoco-mjx",
        ],
    },
    ext_modules=[CMakeExtension("direct_server")],
    cmdclass={
        "build_py": BuildPyCommand,
        "build_ext": BuildCMakeExtension,
        "generate_proto_grpc": GenerateProtoGrpcCommand,
        "copy_direct_server_binary": CopyDirectServerBinaryCommand,
        "copy_task_assets": CopyTaskAssetsCommand,
    },
    package_data={
        "": [
            "mjpc/direct_server",
            # "mjpc/ui_direct_server",
        ],
    },
)
