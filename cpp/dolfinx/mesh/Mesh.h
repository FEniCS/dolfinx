// Copyright (C) 2006-2020 Anders Logg, Chris Richardson and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Geometry.h"
#include "Topology.h"
#include "cell_types.h"
#include "utils.h"
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/UniqueIdGenerator.h>
#include <dolfinx/common/array2d.h>
#include <string>
#include <utility>

namespace dolfinx
{

namespace fem
{
class CoordinateElement;
}

namespace graph
{
template <typename T>
class AdjacencyList;
}

namespace mesh
{

/// @todo Document fully
///
/// Signature for the cell partitioning function. The function should
/// compute the destination rank for cells currently on this rank.
using CellPartitionFunction
    = std::function<const dolfinx::graph::AdjacencyList<std::int32_t>(
        MPI_Comm comm, int nparts, const dolfinx::mesh::CellType cell_type,
        const dolfinx::graph::AdjacencyList<std::int64_t>& cells,
        dolfinx::mesh::GhostMode ghost_mode)>;

/// Enum for different partitioning ghost modes
enum class GhostMode : int
{
  none,
  shared_facet,
  shared_vertex
};

/// A Mesh consists of a set of connected and numbered mesh topological
/// entities, and geometry data
class Mesh
{
public:
  /// Create a mesh
  /// @param[in] comm MPI Communicator
  /// @param[in] topology Mesh topology
  /// @param[in] geometry Mesh geometry
  template <typename Topology, typename Geometry>
  Mesh(MPI_Comm comm, Topology&& topology, Geometry&& geometry)
      : _topology(std::forward<Topology>(topology)),
        _geometry(std::forward<Geometry>(geometry)), _mpi_comm(comm)
  {
    // Do nothing
  }

  /// Copy constructor
  /// @param[in] mesh Mesh to be copied
  Mesh(const Mesh& mesh) = default;

  /// Move constructor
  /// @param mesh Mesh to be moved.
  Mesh(Mesh&& mesh) = default;

  /// Destructor
  ~Mesh() = default;

  // Assignment operator
  Mesh& operator=(const Mesh& mesh) = delete;

  /// Assignment move operator
  /// @param mesh Another Mesh object
  Mesh& operator=(Mesh&& mesh) = default;

  // TODO: Is there any use for this? In many situations one has to get the
  // topology of a const Mesh, which is done by Mesh::topology_mutable. Note
  // that the python interface (calls Mesh::topology()) may still rely on it.
  /// Get mesh topology
  /// @return The topology object associated with the mesh.
  Topology& topology();

  /// Get mesh topology (const version)
  /// @return The topology object associated with the mesh.
  const Topology& topology() const;

  /// Get mesh topology if one really needs the mutable version
  /// @return The topology object associated with the mesh.
  Topology& topology_mutable() const;

  /// Get mesh geometry
  /// @return The geometry object associated with the mesh
  Geometry& geometry();

  /// Get mesh geometry (const version)
  /// @return The geometry object associated with the mesh
  const Geometry& geometry() const;

  /// Get unique identifier for the mesh
  /// @returns The unique identifier associated with the object
  std::size_t id() const { return _unique_id; }

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm mpi_comm() const;

  /// Name
  std::string name = "mesh";

private:
  // Mesh topology:
  // TODO: This is mutable because of the current memory management within
  // mesh::Topology. It allows to obtain a non-const Topology from a
  // const mesh (via Mesh::topology_mutable()).
  //
  mutable Topology _topology;

  // Mesh geometry
  Geometry _geometry;

  // MPI communicator
  dolfinx::MPI::Comm _mpi_comm;

  // Unique identifier
  std::size_t _unique_id = common::UniqueIdGenerator::id();
};

/// Create a mesh using the default partitioner. This function takes
/// mesh input data that is distributed across processes and creates a
/// @p Mesh, with the cell distribution determined by the default cell
/// partitioner. The default partitioner is based a graph partitioning.
///
/// @param[in] comm The MPI communicator to build the mesh on
/// @param[in] cells The cells on the this MPI rank. Each cell (node in
/// the `AdjacencyList`) is defined by its 'nodes' (using global
/// indices). For lowest order cells this will be just the cell
/// vertices. For higher-order cells, other cells 'nodes' will be
/// included.
/// @param[in] element The coordinate element that describes the
/// geometric mapping for cells
/// @param[in] x The coordinates of mesh nodes
/// @param[in] ghost_mode The requested type of cell ghosting/overlap
/// @return A distributed Mesh.
Mesh create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
                 const fem::CoordinateElement& element,
                 const common::array2d<double>& x, GhostMode ghost_mode);

/// Create a mesh using a provided mesh partitioning function
Mesh create_mesh(MPI_Comm comm, const graph::AdjacencyList<std::int64_t>& cells,
                 const fem::CoordinateElement& element,
                 const common::array2d<double>& x, GhostMode ghost_mode,
                 const CellPartitionFunction& cell_partitioner);

} // namespace mesh
} // namespace dolfinx
