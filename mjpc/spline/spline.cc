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
#include <cstddef>
#include <utility>
#include <vector>

#include <absl/log/check.h>
#include <absl/log/log.h>
#include <absl/types/span.h>

namespace mjpc::spline {

TimeSpline::TimeSpline(int dim, int initial_capacity) : dim_(dim) {
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
  ConstNode n = NodeAt(lower - times_.begin());
  std::copy(n.values().begin(), n.values().end(), values.begin());
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
  last_node--;

  int nodes_to_remove = last_node - times_.begin();

  times_.erase(times_.begin(), last_node);
  values_begin_ += dim_ * nodes_to_remove;
  if (values_begin_ >= values_.size()) {
    values_begin_ -= values_.size();
    CHECK_LE(values_begin_, values_.size());
  }
  return nodes_to_remove;
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
}  // namespace mjpc::spline
