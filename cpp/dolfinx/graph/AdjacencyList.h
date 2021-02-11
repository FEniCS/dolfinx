// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cassert>
#include <dolfinx/common/span.hpp>
#include <numeric>
#include <sstream>
#include <utility>
#include <vector>

namespace dolfinx::graph
{

/// Construct adjacency list data for a problem with a fixed number of
/// links (edges) for each node
/// @param [in] matrix Two-dimensional array of adjacency data where
/// matrix(i, j) is the jth neighbor of the ith node
/// @return Adjacency list data and offset array
template <typename Container>
auto create_adjacency_data(const Container& matrix)
{
  using T = typename Container::value_type;

  std::vector<T> data(matrix.size());
  std::vector<std::int32_t> offset(matrix.rows() + 1, 0);

  for (std::size_t i = 0; i < std::size_t(matrix.rows()); ++i)
  {
    for (std::size_t j = 0; j < std::size_t(matrix.cols()); ++j)
      data[i * matrix.cols() + j] = matrix(i, j);
    offset[i + 1] = offset[i] + matrix.cols();
  }
  return std::pair(std::move(data), std::move(offset));
}

/// This class provides a static adjacency list data structure. It is
/// commonly used to store directed graphs. For each node in the
/// contiguous list of nodes [0, 1, 2, ..., n) it stores the connected
/// nodes. The representation is strictly local, i.e. it is not parallel
/// aware.
template <typename T>
class AdjacencyList
{
public:
  /// Construct trivial adjacency list where each of the n nodes is
  /// connected to itself
  /// @param [in] n Number of nodes
  explicit AdjacencyList(const std::int32_t n) : _array(n), _offsets(n + 1)
  {
    std::iota(_array.begin(), _array.end(), 0);
    std::iota(_offsets.begin(), _offsets.end(), 0);
  }

  /// Construct adjacency list from arrays of data
  /// @param [in] data Adjacency array
  /// @param [in] offsets The index to the adjacency list in the data
  /// array for node i
  template <
      typename U, typename V,
      typename = std::enable_if_t<
          std::is_same<std::vector<T>, std::decay_t<U>>::value
          && std::is_same<std::vector<std::int32_t>, std::decay_t<V>>::value>>
  AdjacencyList(U&& data, V&& offsets)
      : _array(std::forward<U>(data)), _offsets(std::forward<V>(offsets))
  {
    assert(_offsets.back() == (std::int32_t)_array.size());
  }

  /// Set all connections for all entities (T is a '2D' container, e.g.
  /// a std::vector<<std::vector<std::size_t>>,
  /// std::vector<<std::set<std::size_t>>, etc)
  /// @param [in] data TODO
  template <typename X>
  explicit AdjacencyList(const std::vector<X>& data)
  {
    // Initialize offsets and compute total size
    _offsets.reserve(data.size() + 1);
    _offsets.push_back(0);
    for (auto row = data.begin(); row != data.end(); ++row)
      _offsets.push_back(_offsets.back() + row->size());

    _array.reserve(_offsets.back());
    for (auto e = data.begin(); e != data.end(); ++e)
      _array.insert(_array.end(), e->begin(), e->end());
  }

  /// Copy constructor
  AdjacencyList(const AdjacencyList& list) = default;

  /// Move constructor
  AdjacencyList(AdjacencyList&& list) = default;

  /// Destructor
  ~AdjacencyList() = default;

  /// Assignment
  AdjacencyList& operator=(const AdjacencyList& list) = default;

  /// Move assignment
  AdjacencyList& operator=(AdjacencyList&& list) = default;

  /// Equality operator
  bool operator==(const AdjacencyList& list) const
  {
    return this->_array == list._array and this->_offsets == list._offsets;
  }

  /// Get the number of nodes
  /// @return The number of nodes
  std::int32_t num_nodes() const { return _offsets.size() - 1; }

  /// Number of connections for given node
  /// @param [in] node Node index
  /// @return The number of outgoing links (edges) from the node
  int num_links(int node) const
  {
    assert((node + 1) < (int)_offsets.size());
    return _offsets[node + 1] - _offsets[node];
  }

  /// Get the links (edges) for given node
  /// @param [in] node Node index
  /// @return Array of outgoing links for the node. The length will be
  /// AdjacencyList:num_links(node).
  tcb::span<T> links(int node)
  {
    return tcb::span(_array.data() + _offsets[node],
                     _offsets[node + 1] - _offsets[node]);
  }

  /// Get the links (edges) for given node (const version)
  /// @param [in] node Node index
  /// @return Array of outgoing links for the node. The length will be
  /// AdjacencyList:num_links(node).
  tcb::span<const T> links(int node) const
  {
    return tcb::span(_array.data() + _offsets[node],
                     _offsets[node + 1] - _offsets[node]);
  }

  /// Return contiguous array of links for all nodes (const version)
  const std::vector<T>& array() const { return _array; }

  /// Return contiguous array of links for all nodes
  std::vector<T>& array() { return _array; }

  /// Offset for each node in array() (const version)
  const std::vector<std::int32_t>& offsets() const { return _offsets; }

  /// Copy of the Adjacency List if the specified type is different from the
  /// current type, ele return a reference.
  template <typename X>
  decltype(auto) as_type() const
  {
// Workaround for Intel compler bug, see
// https://community.intel.com/t5/Intel-C-Compiler/quot-if-constexpr-quot-and-quot-missing-return-statement-quot-in/td-p/1154551
#ifdef __INTEL_COMPILER
#pragma warning(disable : 1011)
#endif

    if constexpr (std::is_same<X, T>::value)
      return *this;
    else
    {
      return graph::AdjacencyList<X>(
          std::vector<X>(_array.begin(), _array.end()), this->_offsets);
    }
  }

  /// Return informal string representation (pretty-print)
  std::string str() const
  {
    std::stringstream s;
    s << "<AdjacencyList> with " + std::to_string(this->num_nodes()) + " nodes"
      << std::endl;
    for (std::size_t e = 0; e < _offsets.size() - 1; ++e)
    {
      s << "  " << e << ": [";
      for (auto link : this->links(e))
        s << link << " ";
      s << "]" << std::endl;
    }
    return s.str();
  }

private:
  // Connections for all entities stored as a contiguous array
  std::vector<T> _array;

  // Position of first connection for each entity (using local index)
  std::vector<std::int32_t> _offsets;
};

/// Construct an adjacency list from array of data for a graph with
/// constant degree (valency). A constant degree graph has the same
/// number of edges for every node.
/// @param [in] data Adjacency array
/// @param [in] degree The number of (outgoing) edges for each node
/// @return An adjacency list
template <typename T, typename U>
AdjacencyList<T> build_adjacency_list(U&& data, int degree)
{
  // using T = typename U::value_type;
  assert(data.size() % degree == 0);
  std::vector<std::int32_t> offsets(data.size() / degree + 1, 0);
  for (std::size_t i = 1; i < offsets.size(); ++i)
    offsets[i] = offsets[i - 1] + degree;
  return AdjacencyList<T>(std::forward<U>(data), std::move(offsets));
}

} // namespace dolfinx::graph
