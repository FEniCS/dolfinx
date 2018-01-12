// Copyright (C) 2006-2016 Anders Logg
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

#pragma once

#include <memory>
#include <string>
#include <utility>

#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include <dolfin/fem/fem_utils.h>
#include "MeshGeometry.h"
#include "MeshConnectivity.h"
#include "MeshTopology.h"

namespace dolfin
{

  class CellType;
  class GenericFunction;
  class LocalMeshData;
  class MeshEntity;
  class Point;
  class SubDomain;
  class BoundingBoxTree;

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

  class Mesh : public Variable
  {
  public:

    /// Create empty mesh
    Mesh(MPI_Comm comm);

    /// Copy constructor.
    ///
    /// @param mesh (Mesh)
    ///         Object to be copied.
    Mesh(const Mesh& mesh);

    /// Create a distributed mesh from local (per process) data.
    ///
    /// @param comm (MPI_Comm)
    ///         MPI communicator for the mesh.
    /// @param local_mesh_data (LocalMeshData)
    ///         Data from which to build the mesh.
    Mesh(MPI_Comm comm, LocalMeshData& local_mesh_data);

    /// Destructor.
    ~Mesh();

    /// Assignment operator
    ///
    /// @param mesh (Mesh)
    ///         Another Mesh object.
    const Mesh& operator=(const Mesh& mesh);

    /// Get number of vertices in mesh.
    ///
    /// @return std::size_t
    ///         Number of vertices.
    ///
    std::int64_t num_vertices() const
    { return _topology.size(0); }

    /// Get number of edges in mesh.
    ///
    /// @return std::size_t
    ///         Number of edges.
    ///
    std::int64_t num_edges() const
    { return _topology.size(1); }

    /// Get number of faces in mesh.
    ///
    /// @return std::size_t
    ///         Number of faces.
    ///
    std::int64_t num_faces() const
    { return _topology.size(2); }

    /// Get number of facets in mesh.
    ///
    /// @return std::size_t
    ///         Number of facets.
    ///
    std::int64_t num_facets() const
    { return _topology.size(_topology.dim() - 1); }

    /// Get number of cells in mesh.
    ///
    /// @return std::size_t
    ///         Number of cells.
    ///
    std::int64_t num_cells() const
    { return _topology.size(_topology.dim()); }

    /// Get number of entities of given topological dimension.
    ///
    /// @param d (std::size_t)
    ///         Topological dimension.
    ///
    /// @return std::size_t
    ///         Number of entities of topological dimension d.
    ///
    std::int64_t num_entities(std::size_t d) const
    { return _topology.size(d); }

    /// Get cell connectivity.
    ///
    /// @return std::vector<unsigned int>&
    ///         Connectivity for all cells.
    ///
    const std::vector<unsigned int>& cells() const
    { return _topology(_topology.dim(), 0)(); }

    /// Get global number of entities of given topological dimension.
    ///
    /// @param dim (std::size_t)
    ///         Topological dimension.
    ///
    /// @return std::int64_t
    ///         Global number of entities of topological dimension d.
    ///
    std::int64_t num_entities_global(std::size_t dim) const
    { return _topology.size_global(dim); }

    /// Get mesh topology.
    ///
    /// @return MeshTopology
    ///         The topology object associated with the mesh.
    MeshTopology& topology()
    { return _topology; }

    /// Get mesh topology (const version).
    ///
    /// @return MeshTopology
    ///         The topology object associated with the mesh.
    const MeshTopology& topology() const
    { return _topology; }

    /// Get mesh geometry.
    ///
    /// @return MeshGeometry
    ///         The geometry object associated with the mesh.
    MeshGeometry& geometry()
    { return _geometry; }

    /// Get mesh geometry (const version).
    ///
    /// @return MeshGeometry
    ///         The geometry object associated with the mesh.
    const MeshGeometry& geometry() const
    { return _geometry; }

    /// Get bounding box tree for mesh. The bounding box tree is
    /// initialized and built upon the first call to this
    /// function. The bounding box tree can be used to compute
    /// collisions between the mesh and other objects. It is the
    /// responsibility of the caller to use (and possibly rebuild) the
    /// tree. It is stored as a (mutable) member of the mesh to enable
    /// sharing of the bounding box tree data structure.
    ///
    /// @return std::shared_ptr<BoundingBoxTree>
    std::shared_ptr<BoundingBoxTree> bounding_box_tree() const;

    /// Get mesh cell type.
    ///
    /// @return CellType&
    ///         The cell type object associated with the mesh.
    CellType& type()
    { dolfin_assert(_cell_type); return *_cell_type; }

    /// Get mesh cell type (const version).
    const CellType& type() const
    { dolfin_assert(_cell_type); return *_cell_type; }

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

    /// Order all mesh entities.
    ///
    /// See also: UFC documentation (put link here!)
    void order();

    /// Check if mesh is ordered according to the UFC numbering convention.
    ///
    /// @return bool
    ///         The return values is true iff the mesh is ordered.
    bool ordered() const;

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
    MPI_Comm mpi_comm() const
    { return _mpi_comm.comm(); }

    /// Ghost mode used for partitioning. Possible values are
    /// same as `parameters["ghost_mode"]`.
    /// WARNING: the interface may change in future without
    /// deprecation; the method is now intended for internal
    /// library use.
    std::string ghost_mode() const;

    // Friend in fem_utils.h
    friend Mesh fem::create_mesh(Function& coordinates);

  private:

    // Friends
    friend class MeshEditor;
    friend class TopologyComputation;
    friend class MeshPartitioning;

    // Mesh topology
    MeshTopology _topology;

    // Mesh geometry
    MeshGeometry _geometry;

    // Bounding box tree used to compute collisions between the mesh
    // and other objects. The tree is initialized to a zero pointer
    // and is allocated and built when bounding_box_tree() is called.
    mutable std::shared_ptr<BoundingBoxTree> _tree;

    // Cell type
    std::unique_ptr<CellType> _cell_type;

    // True if mesh has been ordered
    mutable bool _ordered;

    // MPI communicator
    dolfin::MPI::Comm _mpi_comm;

    // Ghost mode used for partitioning
    std::string _ghost_mode;
  };
}
