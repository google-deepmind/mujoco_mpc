# Copyright 2022 DeepMind Technologies Limited
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

test(direct_force_test)
target_link_libraries(direct_force_test load simulation gmock)

test(direct_parameter_test)
target_link_libraries(direct_parameter_test load simulation gmock)

test(direct_optimize_test)
target_link_libraries(direct_optimize_test load simulation gmock)

test(direct_sensor_test)
target_link_libraries(direct_sensor_test load simulation gmock)

test(direct_trajectory_test)
target_link_libraries(direct_trajectory_test gmock)

test(direct_utilities_test)
target_link_libraries(direct_utilities_test load simulation gmock)
