// Copyright (C) 2006-2013 Anders Logg
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
// Modified by Johan Hoffman 2007
// Modified by Magnus Vikstrøm 2007
// Modified by Garth N. Wells 2007-2015
// Modified by Niclas Jansson 2008
// Modified by Kristoffer Selim 2008
// Modified by Andre Massing 2009-2010
// Modified by Marie E. Rognes 2012
// Modified by Mikael Mortensen 2012
// Modified by Jan Blechta 2013

#ifndef __MESH_H
#define __MESH_H

#include <memory>
#include <string>
#include <utility>

#include <dolfin/ale/MeshDisplacement.h>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/common/MPI.h>
#include <dolfin/common/Variable.h>
#include "MeshData.h"
#include "MeshDomains.h"
#include "MeshGeometry.h"
#include "MeshConnectivity.h"
#include "MeshTopology.h"

namespace dolfin
{
  class BoundaryMesh;
  class CellType;
  class Expression;
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
  /// <tt>
  ///
  /// +--------+-----------+-------------+
  /// | Entity | Dimension | Codimension |
  /// +========+===========+=============+
  /// | Vertex |  0        |             |
  /// +--------+-----------+-------------+
  /// | Edge   |  1        |             |
  /// +--------+-----------+-------------+
  /// | Face   |  2        |             |
  /// +--------+-----------+-------------+
  /// | Facet  |           |      1      |
  /// +--------+-----------+-------------+
  /// | Cell   |           |      0      |
  /// +--------+-----------+-------------+
  ///
  /// </tt>
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

  class Mesh : public Variable, public Hierarchical<Mesh>
  {
  public:

    /// Create empty mesh
    Mesh();

    /// Create empty mesh
    Mesh(MPI_Comm comm);

    /// Copy constructor.
    ///
    /// @param mesh (_Mesh_)
    ///         Object to be copied.
    Mesh(const Mesh& mesh);

    /// Create mesh from data file.
    ///
    /// @param  filename (std::string)
    ///         Name of file to load.
    explicit Mesh(std::string filename);

    /// Create mesh from data file.
    ///
    /// @param comm (MPI_Comm)
    ///         The MPI communicator
    /// @param filename (std::string)
    ///         Name of file to load.
    Mesh(MPI_Comm comm, std::string filename);

    /// Create a distributed mesh from local (per process) data.
    ///
    /// @param comm (MPI_Comm)
    ///         MPI communicator for the mesh.
    /// @param local_mesh_data (_LocalMeshData_)
    ///         Data from which to build the mesh.
    Mesh(MPI_Comm comm, LocalMeshData& local_mesh_data);

    /// Destructor.
    ~Mesh();

    /// Assignment operator
    ///
    /// @param mesh (_Mesh_)
    ///         Another _Mesh_ object.
    const Mesh& operator=(const Mesh& mesh);

    /// Get number of vertices in mesh.
    ///
    /// @return std::size_t
    ///         Number of vertices.
    ///
    std::size_t num_vertices() const
    { return _topology.size(0); }

    /// Get number of edges in mesh.
    ///
    /// @return std::size_t
    ///         Number of edges.
    ///
    std::size_t num_edges() const
    { return _topology.size(1); }

    /// Get number of faces in mesh.
    ///
    /// @return std::size_t
    ///         Number of faces.
    ///
    std::size_t num_faces() const
    { return _topology.size(2); }

    /// Get number of facets in mesh.
    ///
    /// @return std::size_t
    ///         Number of facets.
    ///
    std::size_t num_facets() const
    { return _topology.size(_topology.dim() - 1); }

    /// Get number of cells in mesh.
    ///
    /// @return std::size_t
    ///         Number of cells.
    ///
    std::size_t num_cells() const
    { return _topology.size(_topology.dim()); }

    /// Get number of entities of given topological dimension.
    ///
    /// @param d (std::size_t)
    ///         Topological dimension.
    ///
    /// @return std::size_t
    ///         Number of entities of topological dimension d.
    ///
    std::size_t num_entities(std::size_t d) const
    { return _topology.size(d); }

    /// Get vertex coordinates.
    ///
    /// @return    std::vector<double>&
    ///         Coordinates of all vertices.
    ///
    std::vector<double>& coordinates()
    { return _geometry.x(); }

    /// Return coordinates of all vertices (const version).
    ///
    /// @return    std::vector<double>&
    ///         Coordinates of all vertices.
    ///
    const std::vector<double>& coordinates() const
    { return _geometry.x(); }

    /// Get cell connectivity.
    ///
    /// @return  std::vector<std::size_t>
    ///         Connectivity for all cells.
    ///
    const std::vector<unsigned int>& cells() const
    { return _topology(_topology.dim(), 0)(); }

    /// Get number of local entities of given topological dimension.
    ///
    /// @param  dim (std::size_t)
    ///         Topological dimension.
    ///
    /// @return std::size_t
    ///         Number of local entities of topological dimension d.
    ///
    std::size_t size(std::size_t dim) const
    { return _topology.size(dim); }

    /// Get global number of entities of given topological dimension.
    ///
    /// @param dim (std::size_t)
    ///         Topological dimension.
    ///
    /// @return std::size_t
    ///         Global number of entities of topological dimension d.
    ///
    std::size_t size_global(std::size_t dim) const
    { return _topology.size_global(dim); }

    /// Get mesh topology.
    ///
    /// @return    _MeshTopology_
    ///         The topology object associated with the mesh.
    MeshTopology& topology()
    { return _topology; }

    /// Get mesh topology (const version).
    ///
    /// @return    _MeshTopology_
    ///         The topology object associated with the mesh.
    const MeshTopology& topology() const
    { return _topology; }

    /// Get mesh geometry.
    ///
    ///
    /// @return  _MeshGeometry_
    ///         The geometry object associated with the mesh.
    MeshGeometry& geometry()
    { return _geometry; }

    /// Get mesh geometry (const version).
    ///
    /// @return  _MeshGeometry_
    ///         The geometry object associated with the mesh.
    const MeshGeometry& geometry() const
    { return _geometry; }

    /// Get mesh (sub)domains.
    ///
    /// @return    _MeshDomains_
    ///         The (sub)domains associated with the mesh.
    MeshDomains& domains()
    { return _domains; }

    /// Get mesh (sub)domains.
    const MeshDomains& domains() const
    { return _domains; }

    /// Get bounding box tree for mesh. The bounding box tree is
    /// initialized and built upon the first call to this
    /// function. The bounding box tree can be used to compute
    /// collisions between the mesh and other objects. It is the
    /// responsibility of the caller to use (and possibly rebuild) the
    /// tree. It is stored as a (mutable) member of the mesh to enable
    /// sharing of the bounding box tree data structure.
    ///
    /// @return std::shared_ptr<BoundingBoxTree> BoundingBoxTree
    std::shared_ptr<BoundingBoxTree> bounding_box_tree() const;

    /// Get mesh data.
    ///
    /// @return _MeshData_
    ///         The mesh data object associated with the mesh.
    MeshData& data();

    /// Get mesh data (const version).
    const MeshData& data() const;

    /// Get mesh cell type.
    ///
    /// @return _CellType_
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

    /// Renumber mesh entities by coloring. This function is currently
    /// restricted to renumbering by cell coloring. The cells
    /// (cell-vertex connectivity) and the coordinates of the mesh are
    /// renumbered to improve the locality within each color. It is
    /// assumed that the mesh has already been colored and that only
    /// cell-vertex connectivity exists as part of the mesh.
    /// @return Mesh
    Mesh renumber_by_color() const;

    /// Translate mesh according to a given vector.
    ///
    /// @param  point (Point)
    ///         The vector defining the translation.
    void translate(const Point& point);

    /// Rotate mesh around a coordinate axis through center of mass
    /// of all mesh vertices
    ///
    /// @param    angle (double)
    ///         The number of degrees (0-360) of rotation.
    /// @param    axis (std::size_t)
    ///         The coordinate axis around which to rotate the mesh.
    void rotate(double angle, std::size_t axis=2);

    /// Rotate mesh around a coordinate axis through a given point
    ///
    /// @param    angle (double)
    ///         The number of degrees (0-360) of rotation.
    /// @param    axis (std::size_t)
    ///         The coordinate axis around which to rotate the mesh.
    /// @param    point (_Point_)
    ///         The point around which to rotate the mesh.
    void rotate(double angle, std::size_t axis, const Point& point);

    /// Smooth internal vertices of mesh by local averaging.
    ///
    /// @param    num_iterations (std::size_t)
    ///         Number of iterations to perform smoothing,
    ///         default value is 1.
    void smooth(std::size_t num_iterations=1);

    /// Smooth boundary vertices of mesh by local averaging.
    ///
    /// @param    num_iterations (std::size_t)
    ///         Number of iterations to perform smoothing,
    ///         default value is 1.
    ///
    /// @param    harmonic_smoothing (bool)
    ///         Flag to turn on harmonics smoothing, default
    ///         value is true.
    void smooth_boundary(std::size_t num_iterations=1,
                         bool harmonic_smoothing=true);

    /// Snap boundary vertices of mesh to match given sub domain.
    ///
    /// @param    sub_domain (_SubDomain_)
    ///         A _SubDomain_ object.
    ///
    /// @param    harmonic_smoothing (bool)
    ///         Flag to turn on harmonics smoothing, default
    ///         value is true.
    void snap_boundary(const SubDomain& sub_domain,
                       bool harmonic_smoothing=true);

    /// Color the cells of the mesh such that no two neighboring cells
    /// share the same color. A colored mesh keeps a
    /// CellFunction<std::size_t> named "cell colors" as mesh data which
    /// holds the colors of the mesh.
    ///
    /// @param    coloring_type (std::string)
    ///         Coloring type, specifying what relation makes two
    ///         cells neighbors, can be one of "vertex", "edge" or
    ///         "facet".
    ///
    /// @return    std::vector<std::size_t>
    ///         The colors as a mesh function over the cells of the mesh.
    const std::vector<std::size_t>& color(std::string coloring_type) const;

    /// Color the cells of the mesh such that no two neighboring cells
    /// share the same color. A colored mesh keeps a
    /// CellFunction<std::size_t> named "cell colors" as mesh data which
    /// holds the colors of the mesh.
    ///
    /// @param    coloring_type (std::vector<std::size_t>)
    ///         Coloring type given as list of topological dimensions,
    ///         specifying what relation makes two mesh entities neighbors.
    ///
    /// @return   std::vector<std::size_t>
    ///         The colors as a mesh function over entities of the mesh.
    const std::vector<std::size_t>&
    color(std::vector<std::size_t> coloring_type) const;

    /// Compute minimum cell size in mesh, measured greatest distance
    /// between any two vertices of a cell.
    ///
    /// @return    double
    ///         The minimum cell size. The size is computed using
    ///         Cell::h()
    ///
    double hmin() const;

    /// Compute maximum cell size in mesh, measured greatest distance
    /// between any two vertices of a cell.
    ///
    /// @return    double
    ///         The maximum cell size. The size is computed using
    ///         Cell::h()
    ///
    double hmax() const;

    /// Compute minimum cell inradius.
    ///
    /// @return    double
    ///         The minimum of cells' inscribed sphere radii
    ///
    double rmin() const;

    /// Compute maximum cell inradius.
    ///
    /// @return    double
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
    /// @param    verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// @return    std::string
    ///         An informal representation of the mesh.
    ///
    std::string str(bool verbose) const;

    /// Return cell_orientations (const version)
    ///
    /// @return    std::vector<int>
    ///
    ///         Map from cell index to orientation of cell. Is empty
    ///         if cell orientations have not been computed.
    const std::vector<int>& cell_orientations() const;

    /// Compute and initialize cell_orientations relative to a given
    /// global outward direction/normal/orientation. Only defined if
    /// mesh is orientable.
    ///
    /// @param     global_normal (Expression)
    ///         A global normal direction to the mesh
    void init_cell_orientations(const Expression& global_normal);

    /// Mesh MPI communicator
    /// @return MPI_Comm
    MPI_Comm mpi_comm() const
    { return _mpi_comm; }

    /// Friend in fem_utils.h
    ///    friend Mesh create_mesh(Function&);

  private:

    // Friends
    friend class MeshEditor;
    friend class TopologyComputation;

    // Mesh topology
    MeshTopology _topology;

    // Mesh geometry
    MeshGeometry _geometry;

    // Mesh domains
    MeshDomains _domains;

    // Auxiliary mesh data
    MeshData _data;

    // Bounding box tree used to compute collisions between the mesh
    // and other objects. The tree is initialized to a zero pointer
    // and is allocated and built when bounding_box_tree() is called.
    mutable std::shared_ptr<BoundingBoxTree> _tree;

    // Cell type
    std::unique_ptr<CellType> _cell_type;

    // True if mesh has been ordered
    mutable bool _ordered;

    // Orientation of cells relative to a global direction
    std::vector<int> _cell_orientations;

    // MPI communicator
    MPI_Comm _mpi_comm;

  };
}

#endif
