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

test(batch_band_test)
target_link_libraries(batch_band_test load gmock)

test(batch_covariance_test)
target_link_libraries(batch_covariance_test load gmock)

test(batch_force_test)
target_link_libraries(batch_force_test load gmock)

test(batch_mulmat_test)
target_link_libraries(batch_mulmat_test gmock)

test(batch_optimize_test)
target_link_libraries(batch_optimize_test load gmock)

test(batch_prior_test)
target_link_libraries(batch_prior_test load gmock)

test(batch_sensor_test)
target_link_libraries(batch_sensor_test load gmock)

test(batch_trajectory_test)
target_link_libraries(batch_trajectory_test gmock)

test(batch_utilities_test)
target_link_libraries(batch_utilities_test load gmock)
