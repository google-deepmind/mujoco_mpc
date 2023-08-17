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
get_filename_component(batch_service_proto "./batch.proto" ABSOLUTE)
get_filename_component(batch_service_proto_path "${batch_service_proto}" PATH)
get_filename_component(kalman_service_proto "./kalman.proto" ABSOLUTE)
get_filename_component(kalman_service_proto_path "${kalman_service_proto}" PATH)

# Generated sources
set(agent_service_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/agent.pb.cc")
set(agent_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/agent.pb.h")
set(agent_service_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/agent.grpc.pb.cc")
set(agent_service_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/agent.grpc.pb.h")

set(batch_service_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/batch.pb.cc")
set(batch_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/batch.pb.h")
set(batch_service_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/batch.grpc.pb.cc")
set(batch_service_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/batch.grpc.pb.h")

set(kalman_service_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/kalman.pb.cc")
set(kalman_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/kalman.pb.h")
set(kalman_service_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/kalman.grpc.pb.cc")
set(kalman_service_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/kalman.grpc.pb.h")

message("We need the following for agent/batch_protos:")
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

add_custom_command(
  OUTPUT
  "${batch_service_proto_srcs}"
  "${batch_service_proto_hdrs}"
  "${batch_service_grpc_srcs}"
  "${batch_service_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS
  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
  --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
  -I "${batch_service_proto_path}"
  --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
  "${batch_service_proto}"
  DEPENDS
  "${batch_service_proto}"
)

add_custom_command(
  OUTPUT
  "${kalman_service_proto_srcs}"
  "${kalman_service_proto_hdrs}"
  "${kalman_service_grpc_srcs}"
  "${kalman_service_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS
  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
  --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
  -I "${kalman_service_proto_path}"
  --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
  "${kalman_service_proto}"
  DEPENDS
  "${kalman_service_proto}"
)

# Include generated *.pb.h files
include_directories(${CMAKE_CURRENT_BINARY_DIR}/..)

add_executable(
  agent_server
  agent_server.cc
  ${agent_service_grpc_srcs}
  ${agent_service_grpc_hdrs}
  ${agent_service_proto_srcs}
  ${agent_service_proto_hdrs}
  agent_service.h
  agent_service.cc
  grpc_agent_util.h
  grpc_agent_util.cc)

target_link_libraries(
  agent_server
  # agent_service_grpc_proto
  absl::check
  absl::flags
  absl::flags_parse
  absl::log
  absl::random_random
  absl::status
  absl::strings
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  mujoco::mujoco
  libmjpc
)
target_include_directories(agent_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
message(AGENT_SERVICE_COMPILE_OPTIONS=${AGENT_SERVICE_COMPILE_OPTIONS})
target_compile_options(agent_server PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(agent_server PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})

add_executable(
  ui_agent_server
  ui_agent_server.cc
  ${agent_service_grpc_srcs}
  ${agent_service_grpc_hdrs}
  ${agent_service_proto_srcs}
  ${agent_service_proto_hdrs}
  grpc_agent_util.h
  grpc_agent_util.cc
  ui_agent_service.h
  ui_agent_service.cc)

target_link_libraries(
  ui_agent_server
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  ${_REFLECTION}
  absl::check
  absl::flags
  absl::flags_parse
  absl::log
  absl::random_random
  absl::status
  absl::strings
  glfw
  libmjpc
  lodepng
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
)
target_include_directories(ui_agent_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_compile_options(ui_agent_server PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(ui_agent_server PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})


add_executable(
  batch_server
  batch_server.cc
  ${batch_service_grpc_srcs}
  ${batch_service_grpc_hdrs}
  ${batch_service_proto_srcs}
  ${batch_service_proto_hdrs}
  batch_service.h
  batch_service.cc
)

target_link_libraries(
  batch_server
  # batch_service_grpc_proto
  absl::check
  absl::flags
  absl::flags_parse
  absl::log
  absl::random_random
  absl::status
  absl::strings
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  mujoco::mujoco
  libmjpc
)
target_include_directories(batch_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
message(BATCH_SERVICE_COMPILE_OPTIONS=${BATCH_SERVICE_COMPILE_OPTIONS})
target_compile_options(batch_server PUBLIC ${BATCH_SERVICE_COMPILE_OPTIONS})
target_link_options(batch_server PRIVATE ${BATCH_SERVICE_LINK_OPTIONS})

add_executable(
  kalman_server
  kalman_server.cc
  ${kalman_service_grpc_srcs}
  ${kalman_service_grpc_hdrs}
  ${kalman_service_proto_srcs}
  ${kalman_service_proto_hdrs}
  kalman_service.h
  kalman_service.cc
)

target_link_libraries(
  kalman_server
  # kalman_service_grpc_proto
  absl::check
  absl::flags
  absl::flags_parse
  absl::log
  absl::random_random
  absl::status
  absl::strings
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  mujoco::mujoco
  libmjpc
)

target_include_directories(kalman_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
message(KALMAN_SERVICE_COMPILE_OPTIONS=${KALMAN_SERVICE_COMPILE_OPTIONS})
target_compile_options(kalman_server PUBLIC ${KALMAN_SERVICE_COMPILE_OPTIONS})
target_link_options(kalman_server PRIVATE ${KALMAN_SERVICE_LINK_OPTIONS})
