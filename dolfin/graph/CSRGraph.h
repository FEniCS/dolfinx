// Copyright (C) 2014 Chris Richardson
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:
// Last changed:

#ifndef __CSRGRAPH_H
#define __CSRGRAPH_H

#include <vector>
#include <dolfin/common/MPI.h>

namespace dolfin
{
  /// This class provides a Compressed Sparse Row Graph defined by a
  /// vector containing edges for each node and a vector of offsets
  /// into the edge vector for each node
  ///
  /// In parallel, all nodes must be numbered uniquely across
  /// processes and the connectivity for local nodes, including
  /// connections to off-process nodes are given by edges(), as usual.
  /// The distribution of nodes amongst processes is calculated
  /// automatically at instantiation. The format of the nodes, edges
  /// and distribution is compatible with the formats for ParMETIS and
  /// PT-SCOTCH. See the manuals for these libraries for further
  /// information.

  template<typename T> class CSRGraph
  {

  public:

    explicit CSRGraph(MPI_Comm mpi_comm)
      : _node_offets(1, 0), node_distribution_vec(1, 0), _mpi_comm(mpi_comm)
    {}

    /// Create a CSR Graph from a collection of edges (X is a
    /// container some type, e.g. std::vector<unsigned int> or
    /// std::set<std::size_t>
    template<typename X>
    CSRGraph(MPI_Comm mpi_comm, const std::vector<X>& graph)
      : _node_offets(1, 0), _mpi_comm(mpi_comm)
    {
      // Count number of outgoing edges
      std::size_t num_edges = 0;
      for (auto const &edges : graph)
        num_edges += edges.size();

      // Reserve memory
      _node_offets.reserve(graph.size());
      _edges.reserve(num_edges);

      for (auto const &p : graph)
      {
        _edges.insert(_edges.end(), p.begin(), p.end());
        //_node_offets.push_back(_edges.size());
        _node_offets.push_back(_node_offets.back() + p.size());
      }

      calculate_node_distribution();
    }

    /// Destructor
    ~CSRGraph() {}

    /// Vector containing all edges for all local nodes
    const std::vector<T>& edges() const
    {
      return _edges;
    }

    /// Vector containing index offsets into edges for all local nodes
    /// (plus extra entry marking end)
    const std::vector<T>& nodes() const
    { return _node_offets; }

    /// Number of local edges in graph
    T num_edges() const
    { return _edges.size(); }

    /// Number of local nodes in graph
    T num_nodes() const
    { return _node_offets.size() - 1; }

    /// Total number of nodes in parallel graph
    T num_nodes_global() const
    { return node_distribution_vec.back(); }

    const std::vector<T>& node_distribution() const
    { return node_distribution_vec; }

  private:

    void calculate_node_distribution()
    {
      // Communicate number of nodes between all processors
      MPI::all_gather(_mpi_comm, num_nodes(), node_distribution_vec);

      node_distribution_vec.insert(node_distribution_vec.begin(), 0);
      for (std::size_t i = 1; i != node_distribution_vec.size(); ++i)
        node_distribution_vec[i] += node_distribution_vec[i - 1];
    }

    // Edges in compressed form. Edges for node i
    // are stored in edges[_node_offets[i]:_node_offets[i + 1]]
    std::vector<T> _edges;

    // Offsets of each node into edges (see above)
    std::vector<T> _node_offets;

    // Distribution of nodes across processes in parallel
    std::vector<T> node_distribution_vec;

    // MPI communicator attached to graph
    MPI_Comm _mpi_comm;

  };

}
#endif
