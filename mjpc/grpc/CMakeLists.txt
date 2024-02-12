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

find_package(ZLIB REQUIRED)
set(gRPC_ZLIB_PROVIDER "package" CACHE INTERNAL "")
set(gRPC_BUILD_GRPC_CSHARP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_NODE_PLUGIN OFF)
set(gRPC_BUILD_GRPC_OBJECTIVE_C_PLUGIN OFF)
set(gRPC_BUILD_GRPC_PHP_PLUGIN OFF)
set(gRPC_BUILD_GRPC_RUBY_PLUGIN OFF)
set(RE2_BUILD_TESTING OFF)
set(ZLIB_BUILD_EXAMPLES OFF)

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
  v1.60.1
  TARGETS
  gRPC
)

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
get_filename_component(direct_service_proto "./direct.proto" ABSOLUTE)
get_filename_component(direct_service_proto_path "${direct_service_proto}" PATH)
get_filename_component(filter_service_proto "./filter.proto" ABSOLUTE)
get_filename_component(filter_service_proto_path "${filter_service_proto}" PATH)

# Generated sources
set(agent_service_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/agent.pb.cc")
set(agent_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/agent.pb.h")
set(agent_service_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/agent.grpc.pb.cc")
set(agent_service_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/agent.grpc.pb.h")

set(direct_service_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/direct.pb.cc")
set(direct_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/direct.pb.h")
set(direct_service_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/direct.grpc.pb.cc")
set(direct_service_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/direct.grpc.pb.h")

set(filter_service_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/filter.pb.cc")
set(filter_service_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/filter.pb.h")
set(filter_service_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/filter.grpc.pb.cc")
set(filter_service_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/filter.grpc.pb.h")

message("We need the following for agent/direct_protos:")
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

add_library(agent_service_proto_lib STATIC
  "${agent_service_proto_srcs}"
  "${agent_service_proto_hdrs}"
  "${agent_service_grpc_srcs}"
  "${agent_service_grpc_hdrs}"
)

target_link_libraries(
  agent_service_proto_lib
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
)

add_custom_command(
  OUTPUT
  "${direct_service_proto_srcs}"
  "${direct_service_proto_hdrs}"
  "${direct_service_grpc_srcs}"
  "${direct_service_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS
  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
  --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
  -I "${direct_service_proto_path}"
  --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
  "${direct_service_proto}"
  DEPENDS
  "${direct_service_proto}"
)

add_custom_command(
  OUTPUT
  "${filter_service_proto_srcs}"
  "${filter_service_proto_hdrs}"
  "${filter_service_grpc_srcs}"
  "${filter_service_grpc_hdrs}"
  COMMAND ${_PROTOBUF_PROTOC}
  ARGS
  --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
  --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
  -I "${filter_service_proto_path}"
  --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
  "${filter_service_proto}"
  DEPENDS
  "${filter_service_proto}"
)

# Include generated *.pb.h files
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../..)

add_library(mjpc_agent_service STATIC)
target_sources(
  mjpc_agent_service
  PUBLIC
  agent_service.h
  PRIVATE
  agent_service.cc
  grpc_agent_util.h
  grpc_agent_util.cc
)

target_link_libraries(
  mjpc_agent_service
  PUBLIC
  agent_service_proto_lib
  PRIVATE
  absl::check
  absl::log
  absl::random_random
  absl::status
  absl::strings
  mujoco::mujoco
  libmjpc
)

target_include_directories(mjpc_agent_service
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/../..
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
message(AGENT_SERVICE_COMPILE_OPTIONS=${AGENT_SERVICE_COMPILE_OPTIONS})
target_compile_options(mjpc_agent_service PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(mjpc_agent_service PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})

add_executable(
  agent_server
  agent_server.cc)

target_link_libraries(
  agent_server
  # agent_service_grpc_proto
  mjpc_agent_service
  absl::check
  absl::flags
  absl::flags_parse
  absl::log
  absl::status
  absl::strings
  ${_REFLECTION}
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  mujoco::mujoco
  libmjpc
)

target_include_directories(agent_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
message(AGENT_SERVICE_COMPILE_OPTIONS=${AGENT_SERVICE_COMPILE_OPTIONS})
target_compile_options(agent_server PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(agent_server PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})

add_library(mjpc_ui_agent_service STATIC)
target_sources(
  mjpc_ui_agent_service
  PUBLIC
  ui_agent_service.h
  PRIVATE
  ui_agent_service.cc
  grpc_agent_util.h
  grpc_agent_util.cc
)

target_link_libraries(
  mjpc_ui_agent_service
  PUBLIC
  agent_service_proto_lib
  PRIVATE
  absl::check
  absl::log
  absl::status
  absl::strings
  libmjpc
  mujoco::mujoco
  mujoco::platform_ui_adapter
  threadpool
  Threads::Threads
)

target_include_directories(mjpc_ui_agent_service
  PUBLIC
  ${CMAKE_CURRENT_BINARY_DIR}/../..
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../..
)
target_compile_options(mjpc_ui_agent_service PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(mjpc_ui_agent_service PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})

add_executable(
  ui_agent_server
  ui_agent_server.cc
)

target_link_libraries(
  ui_agent_server
  mjpc_ui_agent_service
  ${_GRPC_GRPCPP}
  ${_PROTOBUF_LIBPROTOBUF}
  ${_REFLECTION}
  absl::flags
  absl::flags_parse
  absl::log
  absl::status
  absl::strings
  glfw
  libmjpc
  mujoco::mujoco
  mujoco::platform_ui_adapter
)

target_include_directories(ui_agent_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_compile_options(ui_agent_server PUBLIC ${AGENT_SERVICE_COMPILE_OPTIONS})
target_link_options(ui_agent_server PRIVATE ${AGENT_SERVICE_LINK_OPTIONS})
if(APPLE)
  target_sources(ui_agent_server PRIVATE ../macos_gui.mm)
  target_link_libraries(ui_agent_server "-framework Cocoa")
endif()

add_executable(
  direct_server
  direct_server.cc
  ${direct_service_grpc_srcs}
  ${direct_service_grpc_hdrs}
  ${direct_service_proto_srcs}
  ${direct_service_proto_hdrs}
  direct_service.h
  direct_service.cc
)

target_link_libraries(
  direct_server
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
target_include_directories(direct_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
message(DIRECT_SERVICE_COMPILE_OPTIONS=${DIRECT_SERVICE_COMPILE_OPTIONS})
target_compile_options(direct_server PUBLIC ${DIRECT_SERVICE_COMPILE_OPTIONS})
target_link_options(direct_server PRIVATE ${DIRECT_SERVICE_LINK_OPTIONS})

add_executable(
  filter_server
  filter_server.cc
  ${filter_service_grpc_srcs}
  ${filter_service_grpc_hdrs}
  ${filter_service_proto_srcs}
  ${filter_service_proto_hdrs}
  filter_service.h
  filter_service.cc
)

target_link_libraries(
  filter_server
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
target_include_directories(filter_server PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
message(FILTER_SERVICE_COMPILE_OPTIONS=${FILTER_SERVICE_COMPILE_OPTIONS})
target_compile_options(filter_server PUBLIC ${FILTER_SERVICE_COMPILE_OPTIONS})
target_link_options(filter_server PRIVATE ${FILTER_SERVICE_LINK_OPTIONS})
