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

namespace dolfin
{

  // Forward declarations

  /// This class provides a Compressed Sparse Row Graph
  /// defined by a vector containing edges for each node
  /// and a vector of offsets into the edge vector for each node

  template <typename T> class CSRGraph
  {

  public:

    explicit CSRGraph(MPI_Comm mpi_comm) : node_vec(1, 0), _mpi_comm(mpi_comm)
    {
    }

    /// Create a CSR Graph from a collection of edges
    CSRGraph(MPI_Comm mpi_comm, const std::vector<std::vector<T> >& graph)
      : node_vec(1, 0), _mpi_comm(mpi_comm)
    {
      for (auto const &p: graph)
        append(p);
    }

    /// Create a CSR Graph from a collection of edges
    CSRGraph(MPI_Comm mpi_comm, const std::vector<std::set<std::size_t> >& graph)
      : node_vec(1, 0), _mpi_comm(mpi_comm)
    {
      for (auto const &p: graph)
      {
        edge_vec.insert(edge_vec.end(), p.begin(), p.end());
        node_vec.push_back(edge_vec.size());
      }
    }

    ~CSRGraph()
    {
    }

    /// Vector containing all edges for all local nodes
    const std::vector<T>& edges() const
    {
      return edge_vec;
    }

    /// Vector containing index offsets into edges for
    /// all local nodes (plus extra entry marking end)
    const std::vector<T>& nodes() const
    {
      return node_vec;
    }

    /// Number of local edges in graph
    std::size_t num_edges()
    {
      return edge_vec.size();
    }

    /// Number of local nodes in graph
    std::size_t num_nodes()
    {
      return node_vec.size() - 1;
    }

    /// Append an extra node (with all edges) on the graph
    void append(const std::vector<T>& edge_row)
    {
      edge_vec.insert(edge_vec.end(), edge_row.begin(), edge_row.end());
      node_vec.push_back(edge_vec.size());
    }

  private:

    // Edges in compressed form. Edges for node i
    // are stored in edges[node_vec[i]:node_vec[i + 1]]
    std::vector<T> edge_vec;

    // Offsets of each node into edges (see above)
    std::vector<T> node_vec;

    // Distribution of nodes across processes in parallel (optional)
    std::vector<T> node_distribution;

    MPI_Comm _mpi_comm;
  };

}
#endif
