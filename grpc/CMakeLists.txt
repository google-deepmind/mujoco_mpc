# Copyright 2023 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

find_package(Threads REQUIRED)

set(BUILD_SHARED_LIBS_OLD ${BUILD_SHARED_LIBS})
set(BUILD_SHARED_LIBS
  OFF
  CACHE INTERNAL "Build SHARED libraries"
)

findorfetch(
  USE_SYSTEM_PACKAGE
  OFF
  PACKAGE_NAME
  gRPC
  LIBRARY_NAME
  gRPC
  GIT_REPO
  https://github.com/grpc/grpc
  GIT_TAG
  v1.53.0
  TARGETS
  gRPC
)


find_package(ZLIB REQUIRED)
set(gRPC_ZLIB_PROVIDER "package" CACHE INTERNAL "")

set(_PROTOBUF_LIBPROTOBUF libprotobuf)
set(_REFLECTION grpc++_reflection)
set(_PROTOBUF_PROTOC $<TARGET_FILE:protoc>)
set(_GRPC_GRPCPP grpc++)
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:grpc_cpp_plugin>)

set(BUILD_SHARED_LIBS
  ${BUILD_SHARED_LIBS_OLD}
  CACHE BOOL "Build SHARED libraries" FORCE
)
unset(BUILD_SHARED_LIBS_OLD)

get_filename_component(agent_service_proto "./agent.proto" ABSOLUTE)
get_filename_component(agent_service_proto_path "${agent_service_proto}" PATH)

# Generated sources
set(agent_service_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/agent.pb.cc")
set(agent_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/agent.pb.h")
set(agent_service_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/agent.grpc.pb.cc")
set(agent_service_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/agent.grpc.pb.h")

message("We need the following for agent_protos:")
message(_GRPC_CPP_PLUGIN_EXECUTABLE${_GRPC_CPP_PLUGIN_EXECUTABLE})

add_custom_command(
  OUTPUT
  "${agent_service_proto_srcs}"
  "${agent_service_proto_hdrs}"
  "${agent_service_grpc_srcs}"
  "${agent_service_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS
  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
  --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
  -I "${agent_service_proto_path}"
  --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
  "${agent_service_proto}"
  DEPENDS
  "${agent_service_proto}"
)

# Include generated *.pb.h files
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)

add_executable(
  agent_service
  agent_service.cc
  ${agent_service_grpc_srcs}
  ${agent_service_grpc_hdrs}
  ${agent_service_proto_srcs}
  ${agent_service_proto_hdrs}
  agent_service_impl.h
  agent_service_impl.cc)

target_link_libraries(
  agent_service
  # agent_service_grpc_proto
  absl::random_random
  absl::status
  absl::flags
  absl::flags_parse
  absl::log
  absl::strings
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  mujoco::mujoco
  libmjpc
)
target_include_directories(agent_service PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
message(MJPC_COMPILE_OPTIONS=${MJPC_COMPILE_OPTIONS})
target_compile_options(agent_service PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(agent_service PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})
