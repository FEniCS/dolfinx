// Copyright (C) 2006-2019 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <boost/functional/hash.hpp>
#include <cassert>
#include <dolfinx/common/span.hpp>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>

namespace dolfinx::graph
{

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

  /// Construct adjacency list from arrays of data (Eigen data types)
  /// @param [in] data Adjacency array
  /// @param [in] offsets The index to the adjacency list in the data
  ///   array for node i
  template <typename U, typename V,
            std::enable_if_t<std::is_base_of<Eigen::EigenBase<std::decay_t<V>>,
                                             std::decay_t<V>>::value,
                             int> = 0>
  AdjacencyList(U& data, V& offsets)
      : _array(data.rows()), _offsets(offsets.rows())
  {
    for (std::size_t i = 0; i < data.size(); ++i)
      _array[i] = data[i];
    for (std::size_t i = 0; i < offsets.size(); ++i)
      _offsets[i] = offsets[i];
  }

  /// Construct adjacency list from arrays of data  (non-Eigen data
  /// types)
  /// @param [in] data Adjacency array
  /// @param [in] offsets The index to the adjacency list in the data
  ///   array for node i
  template <typename U, typename V,
            std::enable_if_t<!std::is_base_of<Eigen::EigenBase<std::decay_t<V>>,
                                              std::decay_t<V>>::value,
                             int> = 0>
  AdjacencyList(U&& data, V&& offsets)
      : _array(std::forward<U>(data)), _offsets(std::forward<V>(offsets))
  {
    assert(_offsets.back() == (std::int32_t)_array.size());
  }

  /// Construct adjacency list for a problem with a fixed number of
  /// links (edges) for each node
  /// @param [in] matrix Two-dimensional array of adjacency data where
  ///   matrix(i, j) is the jth neighbor of the ith node
  explicit AdjacencyList(
      const Eigen::Ref<const Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic,
                                          Eigen::RowMajor>>& matrix)
      : _array(matrix.rows() * matrix.cols()), _offsets(matrix.rows() + 1)
  {
    const std::int32_t num_links = matrix.cols();
    for (std::size_t e = 0; e < _offsets.size(); e++)
      _offsets[e] = e * num_links;

    // NOTE: Do not directly copy data from matrix because it may be a
    // view into a larger array
    for (Eigen::Index i = 0; i < matrix.rows(); ++i)
      for (int j = 0; j < num_links; ++j)
        _array[_offsets[i] + j] = matrix(i, j);
  }

  /// Set all connections for all entities (T is a '2D' container, e.g.
  /// a std::vector<<std::vector<std::size_t>>,
  /// std::vector<<std::set<std::size_t>>, etc)
  /// @param [in] data TODO
  template <typename X>
  explicit AdjacencyList(const std::vector<X>& data) : _offsets(data.size() + 1)
  {
    // Initialize offsets and compute total size
    std::int32_t size = 0;
    for (std::size_t e = 0; e < data.size(); e++)
    {
      _offsets[e] = size;
      size += data[e].size();
    }
    _offsets[data.size()] = size;

    _array.reserve(size);
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

  /// Number of nodes
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

  /// Links (edges) for given node
  /// @param [in] node Node index
  /// @return Array of outgoing links for the node. The length will be
  /// AdjacencyList:num_links(node).
  tcb::span<T> links(int node)
  {
    return tcb::span(_array.data() + _offsets[node],
                     _offsets[node + 1] - _offsets[node]);
  }

  /// Links (edges) for given node (const version)
  /// @param [in] node Node index
  /// @return Array of outgoing links for the node. The length will be
  /// AdjacencyList:num_links(node).
  tcb::span<const T> links(int node) const
  {
    return tcb::span(_array.data() + _offsets[node],
                     _offsets[node + 1] - _offsets[node]);
  }

  /// TODO: attempt to remove
  const std::int32_t* links_ptr(int node) const
  {
    return &_array[_offsets[node]];
  }

  /// Return contiguous array of links for all nodes (const version)
  const std::vector<T>& array() const { return _array; }

  /// Offset for each node in array() (const version)
  const std::vector<std::int32_t>& offsets() const { return _offsets; }

  /// Hash of graph
  std::size_t hash() const
  {
    return boost::hash_range(_array.data(), _array.data() + _array.size());
  }

  /// Copy of the Adjacency List if the specified type is different from the
  /// current type, ele return a reference.
  template <typename X>
  decltype(auto) as_type() const
  {
// Workaround for Intel compler bug, see
// https://community.intel.com/t5/Intel-C-Compiler/quot-if-constexpr-quot-and-quot-missing-return-statement-quot-in/td-p/1154551
#ifdef __INTEL_COMPILER
#pragma warning(disable : 1011) Ì
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
    // for (Eigen::Index e = 0; e < _offsets.size() - 1; e++)
    // {
    //   s << "  " << e << ": "
    //     << _array.segment(_offsets[e], _offsets[e + 1] - _offsets[e])
    //            .transpose()
    //     << std::endl;
    // }

    return s.str();
  }

private:
  // Connections for all entities stored as a contiguous array
  std::vector<T> _array;

  // Position of first connection for each entity (using local index)
  std::vector<std::int32_t> _offsets;
}; // namespace graph
} // namespace dolfinx::graph
