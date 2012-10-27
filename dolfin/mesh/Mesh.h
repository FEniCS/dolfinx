// Copyright (C) 2006-2012 Anders Logg
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
// Modified by Garth N. Wells 2007-2011
// Modified by Niclas Jansson 2008
// Modified by Kristoffer Selim 2008
// Modified by Andre Massing 2009-2010
//
// First added:  2006-05-08
// Last changed: 2012-10-25

#ifndef __MESH_H
#define __MESH_H

#include <string>
#include <utility>
#include <boost/scoped_ptr.hpp>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/common/Hierarchical.h>
#include <dolfin/intersection/IntersectionOperator.h>
#include <dolfin/log/log.h>
#include "MeshData.h"
#include "MeshGeometry.h"
#include "MeshConnectivity.h"
#include "MeshTopology.h"
#include "MeshDomains.h"

namespace dolfin
{

  class CellType;
  class BoundaryMesh;
  class Function;
  class LocalMeshData;
  class MeshEntity;
  template <typename T> class MeshFunction;
  class SubDomain;

  /// A _Mesh_ consists of a set of connected and numbered mesh entities.
  ///
  /// Both the representation and the interface are
  /// dimension-independent, but a concrete interface is also provided
  /// for standard named mesh entities:
  ///
  /// .. tabularcolumns:: |c|c|c|
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

    /// Copy constructor.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         Object to be copied.
    Mesh(const Mesh& mesh);

    /// Create mesh from data file.
    ///
    /// *Arguments*
    ///     filename (std::string)
    ///         Name of file to load.
    explicit Mesh(std::string filename);

    /// Create a distributed mesh from local (per process) data.
    ///
    /// *Arguments*
    ///     local_mesh_data (LocalMeshData)
    ///         Data from which to build the mesh.
    explicit Mesh(LocalMeshData& local_mesh_data);

    /// Destructor.
    ~Mesh();

    /// Assignment operator
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         Another _Mesh_ object.
    const Mesh& operator=(const Mesh& mesh);

    /// Get number of vertices in mesh.
    ///
    /// *Returns*
    ///     uint
    ///         Number of vertices.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint num_vertices() const { return _topology.size(0); }

    /// Get number of locally owned vertices
    /// *Returns*
    ///     uint
    ///         Number of locally owned vertices.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint num_owned_vertices() const;

    std::vector<uint> owned_vertices() const;


    /// Get number of edges in mesh.
    ///
    /// *Returns*
    ///     uint
    ///         Number of edges.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint num_edges() const { return _topology.size(1); }

    /// Get number of faces in mesh.
    ///
    /// *Returns*
    ///     uint
    ///         Number of faces.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint num_faces() const { return _topology.size(2); }

    /// Get number of facets in mesh.
    ///
    /// *Returns*
    ///     uint
    ///         Number of facets.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint num_facets() const { return _topology.size(_topology.dim() - 1); }

    /// Get number of cells in mesh.
    ///
    /// *Returns*
    ///     uint
    ///         Number of cells.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint num_cells() const { return _topology.size(_topology.dim()); }

    /// Get number of entities of given topological dimension.
    ///
    /// *Arguments*
    ///     d (uint)
    ///         Topological dimension.
    ///
    /// *Returns*
    ///     uint
    ///         Number of entities of topological dimension d.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint num_entities(uint d) const { return _topology.size(d); }

    /// Get vertex coordinates.
    ///
    /// *Returns*
    ///     double*
    ///         Coordinates of all vertices.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    double* coordinates() { return _geometry.x(); }

    /// Return coordinates of all vertices (const version).
    const double* coordinates() const { return _geometry.x(); }

    /// Get cell connectivity.
    ///
    /// *Returns*
    ///     uint*
    ///         Connectivity for all cells.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    const uint* cells() const { return _topology(_topology.dim(), 0)(); }

    /// Get number of local entities of given topological dimension.
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         Topological dimension.
    ///
    /// *Returns*
    ///     uint
    ///         Number of local entities of topological dimension d.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint size(uint dim) const { return _topology.size(dim); }

    /// Get global number of entities of given topological dimension.
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         Topological dimension.
    ///
    /// *Returns*
    ///     uint
    ///         Global number of entities of topological dimension d.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    uint size_global(uint dim) const { return _topology.size_global(dim); }

    /// Get mesh topology.
    ///
    /// *Returns*
    ///     _MeshTopology_
    ///         The topology object associated with the mesh.
    MeshTopology& topology() { return _topology; }

    /// Get mesh topology (const version).
    const MeshTopology& topology() const { return _topology; }

    /// Get mesh geometry.
    ///
    /// *Returns*
    ///     _MeshGeometry_
    ///         The geometry object associated with the mesh.
    MeshGeometry& geometry() { return _geometry; }

    /// Get mesh geometry (const version).
    const MeshGeometry& geometry() const { return _geometry; }

    /// Get mesh (sub)domains.
    ///
    /// *Returns*
    ///     _MeshDomains_
    ///         The (sub)domains associated with the mesh.
    MeshDomains& domains() { return _domains; }

    /// Get mesh (sub)domains.
    const MeshDomains& domains() const { return _domains; }

    /// Get intersection operator.
    ///
    /// *Returns*
    ///     _IntersectionOperator_
    ///         The intersection operator object associated with the mesh.
    IntersectionOperator& intersection_operator();

    /// Return intersection operator (const version);
    const IntersectionOperator& intersection_operator() const;

    /// Get mesh data.
    ///
    /// *Returns*
    ///     _MeshData_
    ///         The mesh data object associated with the mesh.
    MeshData& data();

    /// Get mesh data (const version).
    const MeshData& data() const;

    /// Get mesh cell type.
    ///
    /// *Returns*
    ///     _CellType_
    ///         The cell type object associated with the mesh.
    CellType& type() { dolfin_assert(_cell_type); return *_cell_type; }

    /// Get mesh cell type (const version).
    const CellType& type() const { dolfin_assert(_cell_type); return *_cell_type; }

    /// Compute entities of given topological dimension.
    ///
    /// *Arguments*
    ///     dim (uint)
    ///         Topological dimension.
    ///
    /// *Returns*
    ///     uint
    ///         Number of created entities.
    uint init(uint dim) const;

    /// Compute connectivity between given pair of dimensions.
    ///
    /// *Arguments*
    ///     d0 (uint)
    ///         Topological dimension.
    ///
    ///     d1 (uint)
    ///         Topological dimension.
    void init(uint d0, uint d1) const;

    /// Compute all entities and connectivity.
    void init() const;

    /// Clear all mesh data.
    void clear();

    /// Clean out all auxiliary topology data. This clears all
    /// topological data, except the connectivity between cells and
    /// vertices.
    void clean();

    /// Order all mesh entities.
    ///
    /// .. seealso::
    ///
    ///     UFC documentation (put link here!)
    void order();

    /// Check if mesh is ordered according to the UFC numbering convention.
    ///
    /// *Returns*
    ///     bool
    ///         The return values is true iff the mesh is ordered.
    bool ordered() const;

    /// Renumber mesh entities by coloring. This function is currently
    /// restricted to renumbering by cell coloring. The cells
    /// (cell-vertex connectivity) and the coordinates of the mesh are
    /// renumbered to improve the locality within each color. It is
    /// assumed that the mesh has already been colored and that only
    /// cell-vertex connectivity exists as part of the mesh.
    Mesh renumber_by_color() const;

    /// Rotate mesh around a coordinate axis through center of mass
    /// of all mesh vertices
    ///
    /// *Arguments*
    ///     angle (double)
    ///         The number of degrees (0-360) of rotation
    ///     axis (uint)
    ///         The coordinate axis around which to rotate the mesh
    void rotate(double angle, uint axis=2);

    /// Rotate mesh around a coordinate axis through a given point
    ///
    /// *Arguments*
    ///     angle (double)
    ///         The number of degrees (0-360) of rotation
    ///     axis (uint)
    ///         The coordinate axis around which to rotate the mesh
    ///     point (_Point_)
    ///         The point around which to rotate the mesh
    void rotate(double angle, uint axis, const Point& p);

    /// Move coordinates of mesh according to new boundary coordinates.
    ///
    /// *Arguments*
    ///     boundary (_BoundaryMesh_)
    ///         A mesh containing just the boundary cells.
    void move(BoundaryMesh& boundary);

    /// Move coordinates of mesh according to adjacent mesh with common global
    /// vertices.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         A _Mesh_ object.
    void move(Mesh& mesh);

    /// Move coordinates of mesh according to displacement function.
    ///
    /// *Arguments*
    ///     displacement (_Function_)
    ///         A _Function_ object.
    void move(const Function& displacement);

    /// Smooth internal vertices of mesh by local averaging.
    ///
    /// *Arguments*
    ///     num_iterations (uint)
    ///         Number of iterations to perform smoothing,
    ///         default value is 1.
    void smooth(uint num_iterations=1);

    /// Smooth boundary vertices of mesh by local averaging.
    ///
    /// *Arguments*
    ///     num_iterations (uint)
    ///         Number of iterations to perform smoothing,
    ///         default value is 1.
    ///
    ///     harmonic_smoothing (bool)
    ///         Flag to turn on harmonics smoothing, default
    ///         value is true.
    void smooth_boundary(uint num_iterations=1, bool harmonic_smoothing=true);

    /// Snap boundary vertices of mesh to match given sub domain.
    ///
    /// *Arguments*
    ///     sub_domain (_SubDomain_)
    ///         A _SubDomain_ object.
    ///
    ///     harmonic_smoothing (bool)
    ///         Flag to turn on harmonics smoothing, default
    ///         value is true.
    void snap_boundary(const SubDomain& sub_domain, bool harmonic_smoothing=true);

    /// Color the cells of the mesh such that no two neighboring cells
    /// share the same color. A colored mesh keeps a
    /// CellFunction<unsigned int> named "cell colors" as mesh data which
    /// holds the colors of the mesh.
    ///
    /// *Arguments*
    ///     coloring_type (std::string)
    ///         Coloring type, specifying what relation makes two
    ///         cells neighbors, can be one of "vertex", "edge" or
    ///         "facet".
    ///
    /// *Returns*
    ///     MeshFunction<unsigned int>
    ///         The colors as a mesh function over the cells of the mesh.
    const std::vector<unsigned int>& color(std::string coloring_type) const;

    /// Color the cells of the mesh such that no two neighboring cells
    /// share the same color. A colored mesh keeps a
    /// CellFunction<unsigned int> named "cell colors" as mesh data which
    /// holds the colors of the mesh.
    ///
    /// *Arguments*
    ///     coloring_type (std::vector<unsigned int>)
    ///         Coloring type given as list of topological dimensions,
    ///         specifying what relation makes two mesh entinties neighbors.
    ///
    /// *Returns*
    ///     MeshFunction<unsigned int>
    ///         The colors as a mesh function over entities of the mesh.
    const std::vector<unsigned int>& color(std::vector<unsigned int> coloring_type) const;

    /// Compute all cells which are intersected by the given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         A _Point_ object.
    ///
    ///     cells (std::set<uint>)
    ///         A set of indices of all intersected cells.
    void intersected_cells(const Point& point,
                           std::set<uint>& cells) const;

    /// Compute all cells which are intersected by any of a vector of points.
    ///
    /// *Arguments*
    ///     points (std::vector<_Point_>)
    ///         A vector of _Point_ objects.
    ///
    ///     cells (std::set<uint>)
    ///         A set of indices of all intersected cells.
    void intersected_cells(const std::vector<Point>& points,
                           std::set<uint>& cells) const;

    /// Compute all cells which are intersected by the given entity.
    ///
    /// *Arguments*
    ///     entity (_MeshEntity_)
    ///         A _MeshEntity_ object.
    ///
    ///     cells (std::vector<uint>)
    ///         A vector of indices of all intersected cells.
    void intersected_cells(const MeshEntity& entity,
                           std::vector<uint>& cells) const;

    /// Compute all cells which are intersected by any of a vector of entities.
    ///
    /// *Arguments*
    ///     entities (std::vector<_MeshEntity_>)
    ///         A vector of _MeshEntity_ objects.
    ///
    ///     cells (std::set<uint>)
    ///         A vector of indices of all intersected cells.
    void intersected_cells(const std::vector<MeshEntity>& entities,
                           std::set<uint>& cells) const;

    /// Compute all cells which are intersected by the given mesh.
    ///
    /// *Arguments*
    ///     mesh (_Mesh_)
    ///         A _Mesh_ object.
    ///
    ///     cells (std::set<uint>)
    ///         A set of indices of all intersected cells.
    void intersected_cells(const Mesh& mesh,
                           std::set<uint>& cells) const;

    /// Find the cell (if any) containing the given point. If the point
    /// is contained in several cells, the first cell is returned.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         A _Point_ object.
    ///
    /// *Returns*
    ///     int
    ///         The index of the cell containing the point. If no cell
    ///         is found, the return value is -1.
    int intersected_cell(const Point& point) const;

    /// Find the point in the mesh closest to the given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         A _Point_ object.
    ///
    /// *Returns*
    ///     _Point_
    ///         The closest point.
    Point closest_point(const Point& point) const;

    /// Find the cell in the mesh closest to the given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         A _Point_ object.
    ///
    /// *Returns*
    ///     uint
    ///         The index of the closest cell.
    ///
    /// *Example*
    ///     .. code-block:: c++
    ///
    ///         UnitSquare mesh(1, 1);
    ///         Point point(0.0, 2.0);
    ///         info("%d", mesh.closest_cell(point));
    ///
    ///     output::
    ///
    ///         1
    dolfin::uint closest_cell(const Point& point) const;

    /// Find the point and corresponding cell closest to the given point.
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         A _Point_ object.
    ///
    /// *Returns*
    ///     std::pair<_Point_, uint>
    ///         A pair consisting of the closest point and corresponding cell index.
    std::pair<Point, dolfin::uint> closest_point_and_cell(const Point& point) const;

    /// Computes the distance between a given point and the mesh
    ///
    /// *Arguments*
    ///     point (_Point_)
    ///         A _Point_ object.
    ///
    /// *Returns*
    ///     double
    ///         The distance to the mesh.
    double distance(const Point& point) const;

    /// Compute minimum cell diameter.
    ///
    /// *Returns*
    ///     double
    ///         The minimum cell diameter, the diameter is computed as
    ///         two times the circumradius
    ///         (http://mathworld.wolfram.com).
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    double hmin() const;

    /// Compute maximum cell diameter.
    ///
    /// *Returns*
    ///     double
    ///         The maximum cell diameter, the diameter is computed as
    ///         two times the circumradius
    ///         (http://mathworld.wolfram.com).
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    double hmax() const;

    /// Compute hash of mesh, currently based on the has of the mesh
    /// geometry and mesh topology.
    ///
    /// *Returns*
    ///     uint
    ///         A tree-hashed value of the coordinates over all MPI processes
    ///
    std::size_t hash() const;

    /// Informal string representation.
    ///
    /// *Arguments*
    ///     verbose (bool)
    ///         Flag to turn on additional output.
    ///
    /// *Returns*
    ///     std::string
    ///         An informal representation of the mesh.
    ///
    /// *Example*
    ///     .. note::
    ///
    ///         No example code available for this function.
    std::string str(bool verbose) const;

  private:

    // Friends
    friend class MeshEditor;
    friend class TopologyComputation;
    friend class MeshOrdering;
    friend class BinaryFile;

    // Mesh topology
    MeshTopology _topology;

    // Mesh geometry
    MeshGeometry _geometry;

    // Mesh domains
    MeshDomains _domains;

    // Auxiliary mesh data
    MeshData _data;

    // Cell type
    CellType* _cell_type;

    // Intersection detector
    IntersectionOperator _intersection_operator;

    // True if mesh has been ordered
    mutable bool _ordered;

  };

}

#endif
