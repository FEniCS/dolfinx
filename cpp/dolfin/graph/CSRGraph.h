// Copyright (C) 2014 Chris Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/MPI.h>
#include <vector>

namespace dolfin
{
namespace graph
{

/// Compressed Sparse Row graph

/// This class provides a Compressed Sparse Row Graph defined by a
/// vector containing edges for each node and a vector of offsets
/// into the edge vector for each node
///
/// In parallel, all nodes must be numbered from zero on process
/// zero continuously through increasing rank processes. Edges must
/// be defined in terms of the global node numbers. The global node
/// offset of each process is given by node_distribution()
///
/// The format of the nodes, edges and distribution is identical
/// with the formats for ParMETIS and PT-SCOTCH.  See the manuals
/// for these libraries for further information.

template <typename T>
class CSRGraph
{

public:
  /// Access edges individually by using operator()[] to get a node
  /// object
  class node
  {
  public:
    /// Node object, listing a set of outgoing edges
    node(const typename std::vector<T>::const_iterator& begin_it,
         const typename std::vector<T>::const_iterator& end_it)
        : begin_edge(begin_it), end_edge(end_it)
    {
    }

    /// Iterator pointing to beginning of edges
    typename std::vector<T>::const_iterator begin() const { return begin_edge; }

    /// Iterator pointing to beyond end of edges
    typename std::vector<T>::const_iterator end() const { return end_edge; }

    /// Number of outgoing edges for this node
    std::size_t size() const { return (end_edge - begin_edge); }

    /// Access outgoing edge i of this node
    const T& operator[](std::size_t i) const { return *(begin_edge + i); }

  private:
    typename std::vector<T>::const_iterator begin_edge;
    typename std::vector<T>::const_iterator end_edge;
  };

  /// Empty CSR Graph
  CSRGraph() = delete;

  /// Create a CSR Graph from a collection of edges (X is a
  /// container some type, e.g. std::vector<unsigned int> or
  /// std::set<std::size_t>
  template <typename X>
  CSRGraph(MPI_Comm mpi_comm, const std::vector<X>& graph)
      : _node_offsets(1, 0), _mpi_comm(mpi_comm)
  {
    // Count number of outgoing edges (to pre-allocate memory)
    std::size_t num_edges = 0;
    for (auto const& edges : graph)
      num_edges += edges.size();

    // Reserve memory
    _node_offsets.reserve(graph.size());
    _edges.reserve(num_edges);

    // Node-by-node, add outgoing edges
    for (auto const& node_edges : graph)
    {
      _edges.insert(_edges.end(), node_edges.begin(), node_edges.end());
      _node_offsets.push_back(_node_offsets.back() + node_edges.size());
    }

    // Compute node offsets
    calculate_node_distribution();
  }

  /// Create a CSR Graph from ParMETIS style adjacency lists
  CSRGraph(MPI_Comm mpi_comm, const T* xadj, const T* adjncy, std::size_t n)
      : _mpi_comm(mpi_comm)
  {
    _node_offsets.assign(xadj, xadj + n + 1);
    _edges.assign(adjncy, adjncy + xadj[n]);

    // Compute node offsets
    calculate_node_distribution();
  }

  /// Copy constructor
  CSRGraph(const CSRGraph& g) = default;

  /// Move constructor
  CSRGraph(CSRGraph&& g) = default;

  /// Destructor
  ~CSRGraph() = default;

  /// Vector containing all edges for all local nodes
  /// ("adjncy" in ParMETIS)
  const std::vector<T>& edges() const { return _edges; }

  /// Vector containing all edges for all local nodes (non-const)
  /// ("adjncy" in ParMETIS)
  std::vector<T>& edges() { return _edges; }

  /// Return CSRGraph::node object which provides begin() and end()
  /// iterators, also size(), and random-access for the edges of
  /// node i.
  const node operator[](std::size_t i) const
  {
    return node(_edges.begin() + _node_offsets[i],
                _edges.begin() + _node_offsets[i + 1]);
  }

  /// Vector containing index offsets into edges for all local nodes
  /// (plus extra entry marking end) ("xadj" in ParMETIS)
  const std::vector<T>& nodes() const { return _node_offsets; }

  /// Vector containing index offsets into edges for all local nodes
  /// (plus extra entry marking end) ("xadj" in ParMETIS)
  std::vector<T>& nodes() { return _node_offsets; }

  /// Number of local edges in graph
  std::size_t num_edges() const { return _edges.size(); }

  /// Number of edges from node i
  std::size_t num_edges(std::size_t i) const
  {
    assert(i < size());
    return (_node_offsets[i + 1] - _node_offsets[i]);
  }

  /// Number of local nodes in graph
  std::size_t size() const { return _node_offsets.size() - 1; }

  /// Total (global) number of nodes in parallel graph
  T size_global() const { return _node_distribution.back(); }

  /// Return number of nodes (offset) on each process
  const std::vector<T>& node_distribution() const { return _node_distribution; }

  /// Return number of nodes (offset) on each process (non-const)
  std::vector<T>& node_distribution() { return _node_distribution; }

private:
  // Compute offset of number of nodes on each process
  void calculate_node_distribution()
  {
    // Communicate number of nodes between all processors
    const T num_nodes = size();
    MPI::all_gather(_mpi_comm.comm(), num_nodes, _node_distribution);

    _node_distribution.insert(_node_distribution.begin(), 0);
    for (std::size_t i = 1; i != _node_distribution.size(); ++i)
      _node_distribution[i] += _node_distribution[i - 1];
  }

  // Edges in compressed form. Edges for node i are stored in
  // _edges[_node_offsets[i]:_node_offsets[i + 1]]
  std::vector<T> _edges;

  // Offsets of each node into edges (see above) corresponding to
  // the nodes on this process (see below)
  std::vector<T> _node_offsets;

  // Distribution of nodes across processes in parallel i.e. the
  // range of nodes stored on process j is
  // _node_distribution[j]:_node_distribution[j+1]
  std::vector<T> _node_distribution;

  // MPI communicator attached to graph
  dolfin::MPI::Comm _mpi_comm;
};
}
}