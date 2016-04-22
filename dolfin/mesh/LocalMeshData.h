// Copyright (C) 2008 Ola Skavhaug
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
// Modified by Anders Logg, 2008-2009.
//
// First added:  2008-11-28
// Last changed: 2013-02-18
//
// Modified by Anders Logg, 2008-2009.
// Modified by Kent-Andre Mardal, 2011.

#ifndef __LOCAL_MESH_DATA_H
#define __LOCAL_MESH_DATA_H

#include <cstdint>
#include <map>
#include <vector>
#include <boost/multi_array.hpp>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include "CellType.h"

namespace dolfin
{

  class Mesh;

  /// This class stores mesh data on a local processor corresponding
  /// to a portion of a (larger) global mesh.
  ///
  /// Note that the data stored in this class does typically not
  /// correspond to a topologically connected mesh; it merely stores a
  /// list of vertex coordinates, a list of cell-vertex mappings and a
  /// list of global vertex numbers for the locally stored vertices.
  ///
  /// It is typically used for parsing meshes in parallel from mesh
  /// XML files. After local mesh data has been parsed on each
  /// processor, a subsequent repartitioning takes place: first a
  /// geometric partitioning of the vertices followed by a
  /// redistribution of vertex and cell data, and then a topological
  /// partitioning again followed by redistribution of vertex and cell
  /// data, at that point corresponding to topologically connected
  /// meshes instead of local mesh data.

  class LocalMeshData : public Variable
  {
  public:

    /// Create empty local mesh data
    explicit LocalMeshData(const MPI_Comm mpi_comm);

    /// Create local mesh data for given mesh
    explicit LocalMeshData(const Mesh& mesh);

    /// Destructor
    ~LocalMeshData();

    /// Check that all essential data has been intialized, and throw error
    /// if there is a problem
    void check() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    // Clear all data
    void clear();

    // Copy data from mesh
    void extract_mesh_data(const Mesh& mesh);

    // Broadcast mesh data from main process (used when Mesh is created
    // on one process)
    void broadcast_mesh_data(const MPI_Comm mpi_comm);

    // Receive mesh data from main process
    void receive_mesh_data(const MPI_Comm mpi_comm);

    // Unpack received vertex coordinates
    void unpack_vertex_coordinates(const std::vector<double>& values);

    // Unpack received cell vertices
    void unpack_cell_vertices(const std::vector<std::int64_t>& values);

    // Coordinates for all vertices stored on local processor
    boost::multi_array<double, 2> vertex_coordinates;

    // Global vertex indices for all vertices stored on local processor
    std::vector<std::int64_t> vertex_indices;

    // Global vertex indices for all cells stored on local processor
    boost::multi_array<std::int64_t, 2> cell_vertices;

    // Global cell numbers for all cells stored on local processor
    std::vector<std::int64_t> global_cell_indices;

    // Optional process owner for each cell in global_cell_indices
    std::vector<int> cell_partition;

    // Optional weight for each cell for partitioning
    std::vector<std::size_t> cell_weight;

    // Global number of vertices
    std::int64_t num_global_vertices;

    // Global number of cells
    std::int64_t num_global_cells;

    // Number of vertices per cell
    int num_vertices_per_cell;

    // Geometrical dimension
    int gdim;

    // Topological dimension
    int tdim;

    // Cell type
    // FIXME: this should replace the need for num_vertices_per_cell and tdim
    CellType::Type cell_type;

    // Mesh domain data [dim](line, (cell_index, local_index, value))
    std::map<std::size_t, std::vector<std::pair<std::pair<std::size_t,
      std::size_t>, std::size_t>>> domain_data;

    // Return MPI communicator
    MPI_Comm mpi_comm() const
    { return _mpi_comm; }

  private:

    // MPI communicator
    MPI_Comm _mpi_comm;

  };

}

#endif
