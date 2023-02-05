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

test(agent_test)
target_link_libraries(agent_test load gmock)

test(cost_derivatives_test)
target_link_libraries(cost_derivatives_test finite_difference gmock)

test(linear_solve_test)
target_link_libraries(linear_solve_test gmock)

test(norm_test)
target_link_libraries(norm_test finite_difference gmock)

test(rollout_test)
target_link_libraries(rollout_test load gmock)

test(threadpool_test)
target_link_libraries(threadpool_test threadpool gmock)

test(trajectory_test)
target_link_libraries(trajectory_test gmock)

test(utilities_test)
target_link_libraries(utilities_test load threadpool gmock)
