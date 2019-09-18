// Copyright (C) 2006-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "cell_types.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/UniqueIdGenerator.h>
#include <dolfin/common/types.h>
#include <memory>
#include <string>
#include <utility>

namespace dolfin
{

namespace function
{
class Function;
}

namespace mesh
{
class CoordinateDofs;
class Geometry;
enum class GhostMode : int;
class MeshEntity;
class Topology;

/// A _Mesh_ consists of a set of connected and numbered mesh entities.
///
/// Both the representation and the interface are
/// dimension-independent, but a concrete interface is also provided
/// for standard named mesh entities:
///
/// | Entity | Dimension | Codimension  |
/// | ------ | --------- | ------------ |
/// | Vertex |  0        |              |
/// | Edge   |  1        |              |
/// | Face   |  2        |              |
/// | Facet  |           |      1       |
/// | Cell   |           |      0       |
///
/// When working with mesh iterators, all entities and connectivity
/// are precomputed automatically the first time an iterator is
/// created over any given topological dimension or connectivity.
///
/// Note that for efficiency, only entities of dimension zero
/// (vertices) and entities of the maximal dimension (cells) exist
/// when creating a _Mesh_. Other entities must be explicitly created
/// by calling init(). For example, all edges in a mesh may be
/// created by a call to mesh.init(1). Similarly, connectivities
/// such as all edges connected to a given vertex must also be
/// explicitly created (in this case by a call to mesh.create_connectivity(0,
/// 1)).

class Mesh
{
public:
  // FIXME: What about global vertex indices?
  // FIXME: Be explicit in passing geometry degree/type
  /// Construct a Mesh from topological and geometric data.
  ///
  /// In parallel, geometric points must be arranged in global index
  /// order across processes, starting from 0 on process 0, and must not
  /// be duplicated. The points will be redistributed to the processes
  /// that need them.
  ///
  /// Cells should be listed only on the processes they appear on, i.e.
  /// mesh partitioning should be performed on the topology data before
  /// calling the Mesh constructor. Ghost cells, if present, must be at
  /// the end of the list of cells, and the number of ghost cells must
  /// be provided.
  ///
  /// @param[in] comm MPI Communicator
  /// @param[in] type Cell type
  /// @param[in] points Array of geometric points, arranged in global
  ///                   index order
  /// @param[in] cells Array of cells (containing the global point
  ///                  indices for each cell)
  /// @param[in] global_cell_indices Array of global cell indices. If
  ///                                not empty, this must be same size
  ///                                as the number of rows in cells. If
  ///                                empty, global cell indices will be
  ///                                constructed, beginning from 0 on
  ///                                process 0.
  /// @param[in] ghost_mode The ghost mode
  /// @param[in] num_ghost_cells Number of ghost cells on this process
  ///                            (must be at end of list of cells)
  Mesh(MPI_Comm comm, mesh::CellType type,
       const Eigen::Ref<const EigenRowArrayXXd> points,
       const Eigen::Ref<const EigenRowArrayXXi64> cells,
       const std::vector<std::int64_t>& global_cell_indices,
       const GhostMode ghost_mode, std::int32_t num_ghost_cells = 0);

  /// Copy constructor
  /// @param[in] mesh Mesh to be copied
  Mesh(const Mesh& mesh);

  /// Move constructor.
  /// @param mesh Mesh to be moved.
  Mesh(Mesh&& mesh);

  /// Destructor
  ~Mesh();

  // Assignment operator
  Mesh& operator=(const Mesh& mesh) = delete;

  /// Assignment move operator
  /// @param mesh Another Mesh object
  Mesh& operator=(Mesh&& mesh) = default;

  /// Get number of entities of given topological dimension
  /// @param[in] d Topological dimension.
  /// @return Number of entities of topological dimension d
  std::int32_t num_entities(int d) const;

  /// Get global number of entities of given topological dimension
  /// @param[in] dim Topological dimension.
  /// @return Global number of entities of topological dimension d
  std::int64_t num_entities_global(std::size_t dim) const;

  /// Get mesh topology
  /// @return The topology object associated with the mesh.
  Topology& topology();

  /// Get mesh topology (const version)
  /// @return The topology object associated with the mesh.
  const Topology& topology() const;

  /// Get mesh geometry
  /// @return The geometry object associated with the mesh
  Geometry& geometry();

  /// Get mesh geometry (const version)
  /// @return The geometry object associated with the mesh
  const Geometry& geometry() const;

  /// Create entities of given topological dimension.
  /// @param[in] dim Topological dimension
  /// @return Number of created entities
  std::size_t create_entities(int dim) const;

  /// Create connectivity between given pair of dimensions, d0 -> d1
  /// @param[in] d0 Topological dimension
  /// @param[in] d1 Topological dimension
  void create_connectivity(std::size_t d0, std::size_t d1) const;

  /// Compute all entities and connectivity
  void create_connectivity_all() const;

  /// Compute global indices for entity dimension dim
  void create_global_indices(std::size_t dim) const;

  /// Clean out all auxiliary topology data. This clears all topological
  /// data, except the connectivity between cells and vertices.
  void clean();

  /// Compute minimum cell size in mesh, measured greatest distance
  /// between any two vertices of a cell.
  /// @return The minimum cell size. The size is computed using
  ///         Cell::h()
  double hmin() const;

  /// Compute maximum cell size in mesh, measured greatest distance
  /// between any two vertices of a cell
  /// @return The maximum cell size. The size is computed using
  ///         Cell::h()
  double hmax() const;

  /// Compute minimum cell inradius
  /// @return double The minimum of cells' inscribed sphere radii
  double rmin() const;

  /// Compute maximum cell inradius
  /// @return The maximum of cells' inscribed sphere radii
  double rmax() const;

  /// Compute hash of mesh, currently based on the has of the mesh
  /// geometry and mesh topology
  /// @return A tree-hashed value of the coordinates over all MPI
  ///         processes
  std::size_t hash() const;

  /// Get unique identifier for the mesh
  /// @returns The unique identifier associated with the object
  std::size_t id() const { return _unique_id; }

  /// Informal string representation
  /// @param[in] verbose Flag to turn on additional output
  /// @return An informal representation of the mesh.
  std::string str(bool verbose) const;

  /// Mesh MPI communicator
  /// @return The communicator on which the mesh is distributed
  MPI_Comm mpi_comm() const;

  /// Ghost mode used for partitioning. Possible values are same as
  /// `parameters["ghost_mode"]`.
  /// WARNING: the interface may change in future without deprecation;
  /// the method is now intended for internal library use.
  mesh::GhostMode get_ghost_mode() const;

  /// Get coordinate dofs for all local cells
  CoordinateDofs& coordinate_dofs();

  /// Get coordinate dofs for all local cells (const version)
  const CoordinateDofs& coordinate_dofs() const;

  /// FIXME: This should be with Geometry
  /// Polynomial degree of the mesh geometry
  std::int32_t degree() const;

  /// Cell type for this Mesh.
  mesh::CellType cell_type;

private:
  // Mesh topology
  std::unique_ptr<Topology> _topology;

  // Mesh geometry
  std::unique_ptr<Geometry> _geometry;

  // FIXME: This should be in geometry!
  // Coordinate dofs
  std::unique_ptr<CoordinateDofs> _coordinate_dofs;

  // FXIME: This shouldn't be here
  // Mesh geometric degree (in Lagrange basis) describing coordinate
  // dofs
  std::int32_t _degree;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;

  // Ghost mode used for partitioning
  GhostMode _ghost_mode;

  // Unique identifier
  std::size_t _unique_id;
};
} // namespace mesh
} // namespace dolfin
