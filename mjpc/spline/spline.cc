// Copyright 2024 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "mjpc/spline/spline.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <utility>
#include <vector>

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/types/span.h>

namespace mjpc::spline {

TimeSpline::TimeSpline(int dim, SplineInterpolation interpolation,
                       int initial_capacity)
    : interpolation_(interpolation), dim_(dim) {
  values_.resize(initial_capacity * dim);  // Reserve space for node values
}

std::size_t TimeSpline::Size() const { return times_.size(); }

TimeSpline::Node TimeSpline::NodeAt(int index) {
  int values_index_ = values_begin_ + index * dim_;
  if (values_index_ >= values_.size()) {
    values_index_ -= values_.size();
    CHECK_LE(values_index_, values_.size());
  }
  return Node(times_[index], values_.data() + values_index_, dim_);
}

TimeSpline::ConstNode TimeSpline::NodeAt(int index) const {
  int values_index_ = values_begin_ + index * dim_;
  if (values_index_ >= values_.size()) {
    values_index_ -= values_.size();
    CHECK_LE(values_index_, values_.size());
  }
  return ConstNode(times_[index], values_.data() + values_index_, dim_);
}

TimeSpline::iterator TimeSpline::begin() {
  return TimeSpline::iterator(this, 0);
}

TimeSpline::iterator TimeSpline::end() {
  return TimeSpline::iterator(this, times_.size());
}

TimeSpline::const_iterator TimeSpline::cbegin() const {
  return TimeSpline::const_iterator(this, 0);
}

TimeSpline::const_iterator TimeSpline::cend() const {
  return TimeSpline::const_iterator(this, times_.size());
}

// Set Interpolation
void TimeSpline::SetInterpolation(SplineInterpolation interpolation) {
  interpolation_ = interpolation;
}

SplineInterpolation TimeSpline::Interpolation() const { return interpolation_; }

int TimeSpline::Dim() const { return dim_; }

// Reserves memory for at least num_nodes. If the spline already contains
// more nodes, does nothing.
void TimeSpline::Reserve(int num_nodes) {
  if (num_nodes * dim_ <= values_.size()) {
    return;
  }
  if (values_begin_ < values_end_ || times_.empty()) {
    // Easy case: just resize the values_ vector and remap the spans if needed,
    // without any further data copies.
    values_.resize(num_nodes * dim_);
  } else {
    std::vector<double> new_values(num_nodes * dim_);
    // Copy all existing values to the start of the new vector
    std::copy(values_.begin() + values_begin_, values_.end(),
              new_values.begin());
    std::copy(values_.begin(), values_.begin() + values_end_,
              new_values.begin() + values_.size() - values_begin_);
    values_ = std::move(new_values);
    values_begin_ = 0;
    values_end_ = times_.size() * dim_;
  }
}

void TimeSpline::Sample(double time, absl::Span<double> values) const {
  CHECK_EQ(values.size(), dim_)
      << "Tried to sample " << values.size()
      << " values, but the dimensionality of the spline is " << dim_;

  if (times_.empty()) {
    std::fill(values.begin(), values.end(), 0.0);
    return;
  }

  auto upper = std::upper_bound(times_.begin(), times_.end(), time);
  if (upper == times_.end()) {
    ConstNode n = NodeAt(upper - times_.begin() - 1);
    std::copy(n.values().begin(), n.values().end(), values.begin());
    return;
  }
  if (upper == times_.begin()) {
    ConstNode n = NodeAt(upper - times_.begin());
    std::copy(n.values().begin(), n.values().end(), values.begin());
    return;
  }

  auto lower = upper - 1;
  double t = (time - *lower) / (*upper - *lower);
  ConstNode lower_node = NodeAt(lower - times_.begin());
  ConstNode upper_node = NodeAt(upper - times_.begin());
  switch (interpolation_) {
    case SplineInterpolation::kZeroSpline:
      std::copy(lower_node.values().begin(), lower_node.values().end(),
                values.begin());
      return;
    case SplineInterpolation::kLinearSpline:
      for (int i = 0; i < dim_; i++) {
        values[i] =
            lower_node.values().at(i) * (1 - t) + upper_node.values().at(i) * t;
      }
      return;
    case SplineInterpolation::kCubicSpline: {
      std::array<double, 4> coefficients =
          CubicCoefficients(time, lower - times_.begin());
      for (int i = 0; i < dim_; i++) {
        double p0 = lower_node.values().at(i);
        double m0 = Slope(lower - times_.begin(), i);
        double m1 = Slope(upper - times_.begin(), i);
        double p1 = upper_node.values().at(i);
        values[i] = coefficients[0] * p0 + coefficients[1] * m0 +
                    coefficients[2] * p1 + coefficients[3] * m1;
      }
      return;
    }
    default:
      CHECK(false) << "Unknown interpolation: " << interpolation_;
  }
}

std::vector<double> TimeSpline::Sample(double time) const {
  std::vector<double> values(dim_);
  Sample(time, absl::MakeSpan(values));
  return values;
}

int TimeSpline::DiscardBefore(double time) {
  // Find the first node that has n.time > time.
  auto last_node = std::upper_bound(times_.begin(), times_.end(), time);
  if (last_node == times_.begin()) {
    return 0;
  }

  // If using cubic interpolation, include not just the last node before `time`,
  // but the one before that.
  int keep_nodes = interpolation_ == SplineInterpolation::kCubicSpline ? 1 : 0;
  last_node--;
  while (last_node != times_.begin() && keep_nodes) {
    last_node--;
    keep_nodes--;
  }
  int nodes_to_remove = last_node - times_.begin();

  times_.erase(times_.begin(), last_node);
  values_begin_ += dim_ * nodes_to_remove;
  if (values_begin_ >= values_.size()) {
    values_begin_ -= values_.size();
    CHECK_LE(values_begin_, values_.size());
  }
  return nodes_to_remove;
}

void TimeSpline::ShiftTime(double start_time) {
  if (times_.empty()) {
    return;
  }
  double shift = start_time - times_[0];
  for (int i = 0; i < times_.size(); i++) {
    times_[i] += shift;
  }
}

void TimeSpline::Clear() {
  times_.clear();
  values_begin_ = 0;
  values_end_ = 0;
  // Don't change capacity_ or reset values_.
}

// Adds a new set of values at the given time. Implementation is only
// efficient if time is later than any previously added nodes.
TimeSpline::Node TimeSpline::AddNode(double time) {
  return AddNode(time, absl::Span<const double>());  // Default empty values
}

TimeSpline::Node TimeSpline::AddNode(double time,
                                     absl::Span<const double> new_values) {
  CHECK(new_values.size() == dim_ || new_values.empty());
  // TODO(nimrod): Implement node insertion in the middle of the spline
  CHECK(times_.empty() || time > times_.back() || time < times_.front())
      << "Adding nodes to the middle of the spline isn't supported.";
  if (times_.size() * dim_ >= values_.size()) {
    Reserve(times_.size() * 2);
  }
  Node new_node;
  if (times_.empty() || time > times_.back()) {
    CHECK_LE(values_end_ + dim_, values_.size());
    times_.push_back(time);
    values_end_ += dim_;
    if (values_end_ >= values_.size()) {
      CHECK_EQ(values_end_, values_.size());
      values_end_ -= values_.size();
    }
    new_node = NodeAt(times_.size() - 1);
  } else {
    CHECK_LT(time, times_.front());
    values_begin_ -= dim_;
    if (values_begin_ < 0) {
      values_begin_ += values_.size();
    }
    CHECK_LE(values_begin_ + dim_, values_.size());
    times_.push_front(time);
    new_node = NodeAt(0);
  }
  if (!new_values.empty()) {
    std::copy(new_values.begin(), new_values.end(), new_node.values().begin());
  } else {
    std::fill(new_node.values().begin(), new_node.values().end(), 0.0);
  }
  return new_node;
}

std::array<double, 4> TimeSpline::CubicCoefficients(
    double time, int lower_node_index) const {
  std::array<double, 4> coefficients;
  int upper_node_index = lower_node_index + 1;
  CHECK(upper_node_index != times_.size())
      << "CubicCoefficients shouldn't be called for boundary conditions.";
  double lower = times_[lower_node_index];
  double upper = times_[upper_node_index];
  double t = (time - lower) / (upper - lower);

  coefficients[0] = 2.0 * t*t*t - 3.0 * t*t + 1.0;
  coefficients[1] =
      (t*t*t - 2.0 * t*t + t) * (upper - lower);
  coefficients[2] = -2.0 * t*t*t + 3 * t*t;
  coefficients[3] = (t*t*t - t*t) * (upper - lower);

  return coefficients;
}

double TimeSpline::Slope(int node_index, int value_index) const {
  ConstNode node = NodeAt(node_index);
  if (node_index == 0) {
    ConstNode next = NodeAt(node_index + 1);
    // one-sided finite-diff
    return (next.values().at(value_index) - node.values().at(value_index)) /
           (next.time() - node.time());
  }
  ConstNode prev = NodeAt(node_index - 1);
  if (node_index == times_.size() - 1) {
    return (node.values().at(value_index) - prev.values().at(value_index)) /
           (node.time() - prev.time());
  }
  ConstNode next = NodeAt(node_index + 1);
  return 0.5 * (next.values().at(value_index) - node.values().at(value_index)) /
             (next.time() - node.time()) +
         0.5 * (node.values().at(value_index) - prev.values().at(value_index)) /
             (node.time() - prev.time());
}
}  // namespace mjpc::spline
