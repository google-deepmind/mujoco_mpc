set(INSAT_TEST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../..)
set(INSAT_TEST_WORKING_DIR ${CMAKE_CURRENT_SOURCE_DIR})

include(GoogleTest)

add_library(loadcopy STATIC ../test/load.h ../test/load.cc)
target_include_directories(loadcopy PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
target_compile_definitions(loadcopy PUBLIC MJSTATIC)
target_link_libraries(loadcopy PUBLIC mujoco::mujoco absl::core_headers absl::strings)
target_include_directories(loadcopy PRIVATE ${mujoco_SOURCE_DIR}/include gmock)

#target_include_directories(opt_test PRIVATE ${INSAT_TEST_INCLUDE} ${mujoco_SOURCE_DIR}/include)
#set_target_properties(opt_test PROPERTIES BUILD_RPATH ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

