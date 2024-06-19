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

#ifndef MJPC_MJPC_SPLINE_SPLINE_H_
#define MJPC_MJPC_SPLINE_SPLINE_H_

#include <array>
#include <cstddef>
#include <deque>
#include <iterator>
#include <type_traits>
#include <vector>

#include <absl/log/check.h>
#include <absl/types/span.h>

namespace mjpc::spline {

enum SplineInterpolation : int {
  kZeroSpline,
  kLinearSpline,
  kCubicSpline,
};


// Represents a spline where values are interpolated based on time.
// Allows updating the spline by adding new future points, or removing old
// nodes.
// This class is not thread safe and requires locking to use.
class TimeSpline {
 public:
  explicit TimeSpline(int dim = 0,
                      SplineInterpolation interpolation = kZeroSpline,
                      int initial_capacity = 1);

  // Copyable, Movable.
  TimeSpline(const TimeSpline& other) = default;
  TimeSpline& operator=(const TimeSpline& other) = default;
  TimeSpline(TimeSpline&& other) = default;
  TimeSpline& operator=(TimeSpline&& other) = default;

  // A view into one spline node in the spline.
  // Template parameter is needed to support both `double` and `const double`
  // views of the data.
  template <typename T>
  class NodeT {
   public:
    NodeT() : time_(0) {};
    NodeT(double time, T* values, int dim)
        : time_(time), values_(values, dim) {}

    // Copyable, Movable.
    NodeT(const NodeT& other) = default;
    NodeT& operator=(const NodeT& other) = default;
    NodeT(NodeT&& other) = default;
    NodeT& operator=(NodeT&& other) = default;

    double time() const { return time_; }

    // Returns a span pointing to the spline values of the node.
    // This function returns a non-const span, to allow spline values to be
    // modified, while the time member and underlying values pointer remain
    // constant.
    absl::Span<T> values() const { return values_; }

   private:
    double time_;
    absl::Span<T> values_;
  };

  using Node = NodeT<double>;
  using ConstNode = NodeT<const double>;

  // Iterator type for TimeSpline.
  // SplineType is TimeSpline or const TimeSpline.
  // NodeType is Node or ConstNode.
  template <typename SplineType, typename NodeType>
  class IteratorT {
   public:
    using iterator_category = std::random_access_iterator_tag;
    using value_type = typename std::remove_cv_t<NodeType>;
    using difference_type = int;
    using pointer = NodeType*;
    using reference = NodeType&;

    IteratorT(SplineType* spline = nullptr, int index = 0)
        : spline_(spline), index_(index) {
      if (spline_ != nullptr && index_ != spline->Size()) {
        node_ = spline->NodeAt(index_);
      }
    }

    // Copyable, Movable.
    IteratorT(const IteratorT& other) = default;
    IteratorT& operator=(const IteratorT& other) = default;
    IteratorT(IteratorT&& other) = default;
    IteratorT& operator=(IteratorT&& other) = default;

    reference operator*() { return node_; }

    pointer operator->() { return &node_; }
    pointer operator->() const { return &node_; }

    IteratorT& operator++() {
      ++index_;
      node_ = index_ == spline_->Size() ? NodeType() : spline_->NodeAt(index_);
      return *this;
    }

    IteratorT operator++(int) {
      IteratorT tmp = *this;
      ++(*this);
      return tmp;
    }

    IteratorT& operator--() {
      --index_;
      node_ = spline_->NodeAt(index_);
      return *this;
    }

    IteratorT operator--(int) {
      IteratorT tmp = *this;
      --(*this);
      return tmp;
    }

    IteratorT& operator+=(difference_type n) {
      if (n != 0) {
        index_ += n;
        node_ =
            index_ == spline_->Size() ? NodeType() : spline_->NodeAt(index_);
      }
      return *this;
    }

    IteratorT& operator-=(difference_type n) { return *this += -n; }

    IteratorT operator+(difference_type n) const {
      IteratorT tmp(*this);
      tmp += n;
      return tmp;
    }

    IteratorT operator-(difference_type n) const {
      IteratorT tmp(*this);
      tmp -= n;
      return tmp;
    }

    friend IteratorT operator+(difference_type n, const IteratorT& it) {
      return it + n;
    }

    friend difference_type operator-(const IteratorT& x, const IteratorT& y) {
      CHECK_EQ(x.spline_, y.spline_)
          << "Comparing iterators from different splines";
      if (x != y) return (x.index_ - y.index_);
      return 0;
    }

    NodeType operator[](difference_type n) const { return *(*this + n); }

    friend bool operator==(const IteratorT& x, const IteratorT& y) {
      return x.spline_ == y.spline_ && x.index_ == y.index_;
    }

    friend bool operator!=(const IteratorT& x, const IteratorT& y) {
      return !(x == y);
    }

    friend bool operator<(const IteratorT& x, const IteratorT& y) {
      CHECK_EQ(x.spline_, y.spline_)
          << "Comparing iterators from different splines";
      return x.index_ < y.index_;
    }

    friend bool operator>(const IteratorT& x, const IteratorT& y) {
      return y < x;
    }

    friend bool operator<=(const IteratorT& x, const IteratorT& y) {
      return !(y < x);
    }

    friend bool operator>=(const IteratorT& x, const IteratorT& y) {
      return !(x < y);
    }

   private:
    SplineType* spline_ = nullptr;
    int index_ = 0;
    NodeType node_;
  };

  using iterator = IteratorT<TimeSpline, Node>;
  using const_iterator = IteratorT<const TimeSpline, ConstNode>;

  // Returns the number of nodes in the spline.
  std::size_t Size() const;


  // Returns the node at the given index, sorted by time. Any calls that mutate
  // the spline will invalidate the Node object.
  Node NodeAt(int index);
  ConstNode NodeAt(int index) const;

  // Returns an iterator that iterates over spline nodes in time order.
  // Callers must not mutate `time`, but can modify values in `values`.
  iterator begin();
  iterator end();
  const_iterator cbegin() const;
  const_iterator cend() const;

  void SetInterpolation(SplineInterpolation interpolation);
  SplineInterpolation Interpolation() const;

  // Returns the dimensionality of interpolation values.
  int Dim() const;

  // Reserves memory for at least num_nodes. If the spline already contains
  // more nodes, does nothing.
  void Reserve(int num_nodes);

  // Interpolates values based on time, writes results to `values`.
  void Sample(double time, absl::Span<double> values) const;
  // Interpolates values based on time, returns a vector of length Dim.
  std::vector<double> Sample(double time) const;

  // Removes any old nodes that have no effect on the values at time `time`.
  // Returns the number of nodes removed.
  int DiscardBefore(double time);

  // Keeps all existing nodes, but shifts the time of the first node to be
  // `start_time`, and all other times are shifted accordingly. No resampling
  // is performed.
  void ShiftTime(double start_time);

  // Removes all existing nodes.
  void Clear();

  // Adds a new set of values at the given time.
  // This class only supports adding nodes with a time later or earlier than
  // all other nodes.
  Node AddNode(double time);
  Node AddNode(double time, absl::Span<const double> values);

 private:
  std::array<double, 4> CubicCoefficients(double time,
                                          int lower_node_index) const;
  double Slope(int node_index, int value_index) const;
  SplineInterpolation interpolation_;

  int dim_;

  // The time values for each node. This is kept sorted.
  std::deque<double> times_;

  // The raw node values. Stored in a ring buffer, which is resized whenever
  // too many nodes are added.
  std::vector<double> values_;

  // The index in values_ for the data of the earliest node.
  int values_begin_ = 0;

  // One past the index in values_ for the end of the data of the last node.
  // If values_end_ == values_begin_, either there's no data (nodes_ is empty),
  // or the values_ buffer is full.
  int values_end_ = 0;
};

}  // namespace mjpc::spline

#endif  // MJPC_MJPC_SPLINE_SPLINE_H_
