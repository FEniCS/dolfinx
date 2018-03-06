// Copyright (C) 2008 Ola Skavhaug
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <boost/multi_array.hpp>
#include <cstdint>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Mesh;

/// This class stores mesh data on a local processor corresponding to a portion
/// of a (larger) global mesh.

/// Note that the data stored in this class does typically not
/// correspond to a topologically connected mesh; it merely stores a
/// list of vertex coordinates, a list of cell-vertex mappings and a
/// list of global vertex numbers for the locally stored vertices.

class LocalMeshData : public common::Variable
{
public:
  /// Create empty local mesh data
  explicit LocalMeshData(const MPI_Comm mpi_comm);

  // Disable copy constructor
  LocalMeshData(const LocalMeshData& data) = delete;

  /// Move constructor
  LocalMeshData(LocalMeshData&& data) = default;

  /// Create local mesh data from a given mesh
  explicit LocalMeshData(const Mesh& mesh);

  /// Destructor
  ~LocalMeshData() = default;

  // Disable assignement operator
  LocalMeshData& operator=(const LocalMeshData& data) = delete;

  /// Check that all essential data has been initialized, and throw error if
  /// there is a problem
  void check() const;

  /// Return informal string representation (pretty-print)
  std::string str(bool verbose) const;

  /// Holder for geometry data
  struct Geometry
  {
    // Constructors
    Geometry() : dim(-1), num_global_vertices(-1) {}
    Geometry(const Geometry& g) = default;
    Geometry(Geometry&& g) = default;

    // Destructor
    ~Geometry() = default;

    // Geometric dimension
    int dim;

    // Global number of vertices
    std::int64_t num_global_vertices;

    // Coordinates for all vertices stored on local processor
    boost::multi_array<double, 2> vertex_coordinates;

    // Global vertex indices for all vertices stored on local processor
    std::vector<std::int64_t> vertex_indices;
  };

  /// Geometry data
  Geometry geometry;

  /// Holder for topology data
  struct Topology
  {
    // Constructor
    Topology() : dim(-1), num_global_cells(-1) {}
    Topology(const Topology& g) = default;
    Topology(Topology&& g) = default;

    // Destructor
    ~Topology() = default;

    // Topological dimension
    int dim;

    // Global number of cells
    std::int64_t num_global_cells;

    // Number of vertices per cell
    int num_vertices_per_cell;

    // Global vertex indices for all cells stored on local processor
    boost::multi_array<std::int64_t, 2> cell_vertices;

    // Global cell numbers for all cells stored on local processor
    std::vector<std::int64_t> global_cell_indices;

    // Optional process owner for each cell in global_cell_indices
    std::vector<int> cell_partition;

    // Optional weight for each cell for partitioning
    std::vector<std::size_t> cell_weight;

    // FIXME: this should replace the need for num_vertices_per_cell
    //        and tdim
    /// Cell type
    mesh::CellType::Type cell_type;
  };

  /// Holder for topology data
  Topology topology;

  /// Return MPI communicator
  MPI_Comm mpi_comm() const { return _mpi_comm.comm(); }

private:
  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;
};
}
}