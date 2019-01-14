// Copyright (C) 2006-2016 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include "CoordinateDofs.h"
#include "MeshConnectivity.h"
#include "MeshGeometry.h"
#include "MeshTopology.h"
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/types.h>
#include <dolfin/fem/utils.h>
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
enum class GhostMode : int;
class MeshEntity;

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
/// explicitly created (in this case by a call to mesh.init(0, 1)).

class Mesh : public common::Variable
{
public:
  /// Construct a Mesh from topological and geometric data.
  ///
  /// In parallel, geometric points must be arranged in global index order
  /// across processes, starting from 0 on process 0, and must not be
  /// duplicated. The points will be redistributed to the processes that need
  /// them.
  ///
  /// Cells should be listed only on the processes they appear on, i.e. mesh
  /// partitioning should be performed on the topology data before calling the
  /// Mesh constructor. Ghost cells, if present, must be at the end of the list
  /// of cells, and the number of ghost cells must be provided.
  ///
  /// @param comm (MPI_Comm)
  ///         MPI Communicator
  /// @param type (CellType::Type)
  ///         Cell type
  /// @param points
  ///         Array of geometric points, arranged in global index order
  /// @param cells
  ///         Array of cells (containing the global point indices for each
  ///         cell)
  /// @param global_cell_indices
  ///         Array of global cell indices. If not empty, this must be same size
  ///         as the number of rows in cells. If empty, global cell indices will
  ///         be constructed, beginning from 0 on process 0.
  /// @param num_ghost_cells
  ///         Number of ghost cells on this process (must be at end of list of
  ///         cells)
  // FIXME: What about global vertex indices?
  // FIXME: Be explicit in passing geometry degree/type
  Mesh(MPI_Comm comm, mesh::CellType::Type type,
       const Eigen::Ref<const EigenRowArrayXXd> points,
       const Eigen::Ref<const EigenRowArrayXXi64> cells,
       const std::vector<std::int64_t>& global_cell_indices,
       const GhostMode ghost_mode, std::uint32_t num_ghost_cells = 0);

  /// Copy constructor.
  ///
  /// @param mesh (Mesh)
  ///         Object to be copied.
  Mesh(const Mesh& mesh);

  /// Move constructor.
  ///
  /// @param mesh (Mesh)
  ///         Object to be moved.
  Mesh(Mesh&& mesh);

  /// Destructor.
  ~Mesh();

  /// Assignment operator
  ///
  /// @param mesh (Mesh)
  ///         Another Mesh object.
  Mesh& operator=(const Mesh& mesh);

  /// Get number of vertices in mesh.
  ///
  /// @return std::size_t
  ///         Number of vertices.
  ///
  std::int64_t num_vertices() const { return _topology.size(0); }

  /// Get number of facets in mesh.
  ///
  /// @return std::size_t
  ///         Number of facets.
  ///
  std::int64_t num_facets() const
  {
    return _topology.size(_topology.dim() - 1);
  }

  /// Get number of cells in mesh.
  ///
  /// @return std::size_t
  ///         Number of cells.
  ///
  std::int64_t num_cells() const { return _topology.size(_topology.dim()); }

  /// Get number of entities of given topological dimension.
  ///
  /// @param d (std::size_t)
  ///         Topological dimension.
  ///
  /// @return std::size_t
  ///         Number of entities of topological dimension d.
  ///
  std::int64_t num_entities(std::size_t d) const { return _topology.size(d); }

  /// Get cell connectivity.
  ///
  /// @return std::vector<std::uint32_t>&
  ///         Connectivity for all cells.
  ///
  Eigen::Ref<const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>> cells() const
  {
    return _topology.connectivity(_topology.dim(), 0).connections();
  }

  /// Get global number of entities of given topological dimension.
  ///
  /// @param dim (std::size_t)
  ///         Topological dimension.
  ///
  /// @return std::int64_t
  ///         Global number of entities of topological dimension d.
  ///
  std::int64_t num_entities_global(std::size_t dim) const
  {
    return _topology.size_global(dim);
  }

  /// Get mesh topology.
  ///
  /// @return MeshTopology
  ///         The topology object associated with the mesh.
  MeshTopology& topology() { return _topology; }

  /// Get mesh topology (const version).
  ///
  /// @return MeshTopology
  ///         The topology object associated with the mesh.
  const MeshTopology& topology() const { return _topology; }

  /// Get mesh geometry.
  ///
  /// @return MeshGeometry
  ///         The geometry object associated with the mesh.
  MeshGeometry& geometry() { return _geometry; }

  /// Get mesh geometry (const version).
  ///
  /// @return MeshGeometry
  ///         The geometry object associated with the mesh.
  const MeshGeometry& geometry() const { return _geometry; }

  /// Get mesh cell type.
  ///
  /// @return CellType&
  ///         The cell type object associated with the mesh.
  mesh::CellType& type()
  {
    assert(_cell_type);
    return *_cell_type;
  }

  /// Get mesh cell type (const version).
  const mesh::CellType& type() const
  {
    assert(_cell_type);
    return *_cell_type;
  }

  /// Compute entities of given topological dimension.
  ///
  /// @param  dim (std::size_t)
  ///         Topological dimension.
  ///
  /// @return std::size_t
  ///         Number of created entities.
  std::size_t init(std::size_t dim) const;

  /// Compute connectivity between given pair of dimensions.
  ///
  /// @param    d0 (std::size_t)
  ///         Topological dimension.
  ///
  /// @param    d1 (std::size_t)
  ///         Topological dimension.
  void init(std::size_t d0, std::size_t d1) const;

  /// Compute all entities and connectivity.
  void init() const;

  /// Compute global indices for entity dimension dim
  void init_global(std::size_t dim) const;

  /// Clean out all auxiliary topology data. This clears all
  /// topological data, except the connectivity between cells and
  /// vertices.
  void clean();

  /// Compute minimum cell size in mesh, measured greatest distance
  /// between any two vertices of a cell.
  ///
  /// @return double
  ///         The minimum cell size. The size is computed using
  ///         Cell::h()
  ///
  double hmin() const;

  /// Compute maximum cell size in mesh, measured greatest distance
  /// between any two vertices of a cell.
  ///
  /// @return double
  ///         The maximum cell size. The size is computed using
  ///         Cell::h()
  ///
  double hmax() const;

  /// Compute minimum cell inradius.
  ///
  /// @return double
  ///         The minimum of cells' inscribed sphere radii
  ///
  double rmin() const;

  /// Compute maximum cell inradius.
  ///
  /// @return double
  ///         The maximum of cells' inscribed sphere radii
  ///
  double rmax() const;

  /// Compute hash of mesh, currently based on the has of the mesh
  /// geometry and mesh topology.
  ///
  /// @return std::size_t
  ///         A tree-hashed value of the coordinates over all MPI processes
  ///
  std::size_t hash() const;

  /// Informal string representation.
  ///
  /// @param verbose (bool)
  ///         Flag to turn on additional output.
  ///
  /// @return std::string
  ///         An informal representation of the mesh.
  ///
  std::string str(bool verbose) const;

  /// Mesh MPI communicator
  /// @return MPI_Comm
  MPI_Comm mpi_comm() const { return _mpi_comm.comm(); }

  /// Ghost mode used for partitioning. Possible values are
  /// same as `parameters["ghost_mode"]`.
  /// WARNING: the interface may change in future without
  /// deprecation; the method is now intended for internal
  /// library use.
  mesh::GhostMode get_ghost_mode() const;

  /// Get coordinate dofs for all local cells
  const CoordinateDofs& coordinate_dofs() const { return _coordinate_dofs; }

  // FIXME: This should be with MeshGeometry
  std::uint32_t degree() const { return _degree; }

private:
  // Cell type
  std::unique_ptr<mesh::CellType> _cell_type;

  // Mesh topology
  MeshTopology _topology;

  // Mesh geometry
  MeshGeometry _geometry;

  // FIXME: This should be in geometry!
  // Coordinate dofs
  CoordinateDofs _coordinate_dofs;

  // FXIME: This shouldn't be here
  // Mesh geometric degree (in Lagrange basis) describing coordinate dofs
  std::uint32_t _degree;

  // MPI communicator
  dolfin::MPI::Comm _mpi_comm;

  // Ghost mode used for partitioning
  GhostMode _ghost_mode;
};
} // namespace mesh
} // namespace dolfin
