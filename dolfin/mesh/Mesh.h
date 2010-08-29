// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman, 2007.
// Modified by Magnus Vikstrøm, 2007.
// Modified by Garth N. Wells, 2007.
// Modified by Niclas Jansson, 2008.
// Modified by Kristoffer Selim, 2008.
// Modified by Andre Massing, 2009-2010.
//
// First added:  2006-05-08
// Last changed: 2010-08-29

#ifndef __MESH_H
#define __MESH_H

#include <string>
#include <utility>

#include <dolfin/common/types.h>
#include <dolfin/common/Variable.h>
#include <dolfin/ale/ALEType.h>
#include "MeshTopology.h"
#include "MeshGeometry.h"
#include "MeshData.h"
#include "IntersectionOperator.h"
#include "CellType.h"

namespace dolfin
{

  template <class T> class MeshFunction;
  class Function;
  class BoundaryMesh;
  class XMLMesh;
  class SubDomain;

  /// A Mesh consists of a set of connected and numbered mesh entities.
  ///
  /// Both the representation and the interface are
  /// dimension-independent, but a concrete interface is also provided
  /// for standard named mesh entities:
  ///
  /// .. tabularcolumns:: |c|c|c|
  ///
  ///     +--------+-----------+-------------+
  ///     | Entity | Dimension | Codimension |
  ///     +========+===========+=============+
  ///     | Vertex |  0        |             |
  ///     +--------+-----------+-------------+
  ///     | Edge   |  1        |             |
  ///     +--------+-----------+-------------+
  ///     | Face   |  2        |             |
  ///     +--------+-----------+-------------+
  ///     | Facet  |           |      1      |
  ///     +--------+-----------+-------------+
  ///     | Cell   |           |        0    |
  ///     +--------+-----------+-------------+
  ///
  /// When working with mesh iterators, all entities and connectivity
  /// are precomputed automatically the first time an iterator is
  /// created over any given topological dimension or connectivity.
  ///
  /// Note that for efficiency, only entities of dimension zero
  /// (vertices) and entities of the maximal dimension (cells) exist
  /// when creating a Mesh. Other entities must be explicitly created
  /// by calling init(). For example, all edges in a mesh may be
  /// created by a call to mesh.init(1). Similarly, connectivities
  /// such as all edges connected to a given vertex must also be
  /// explicitly created (in this case by a call to mesh.init(0, 1)).

  class Mesh : public Variable
  {
  public:

    /// Create empty mesh
    Mesh();

    /// Copy constructor.
    ///
    /// *Arguments*
    ///     mesh
    ///         A Mesh object.
    Mesh(const Mesh& mesh);

    /// Create mesh from data file.
    ///
    /// *Arguments*
    ///     filename
    ///         A string, name of file to load.
    explicit Mesh(std::string filename);

    /// Destructor.
    ~Mesh();

    /// Assignment operator
    ///
    /// *Arguments*
    ///     mesh
    ///         A :cpp:class:`Mesh` object.
    const Mesh& operator=(const Mesh& mesh);

    /// Get number of vertices in mesh.
    ///
    /// *Returns*
    ///     integer
    ///         Number of vertices.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.num_vertices()
    ///     9
    uint num_vertices() const { return _topology.size(0); }

    /// Get number of edges in mesh.
    ///
    /// *Returns*
    ///     integer
    ///         Number of edges.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.num_edges()
    ///     0
    ///     >>> mesh.init(1)
    ///     16
    ///     >>> mesh.num_edges()
    ///     16
    uint num_edges() const { return _topology.size(1); }

    /// Get number of faces in mesh.
    ///
    /// *Returns*
    ///     integer
    ///         Number of faces.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.num_faces()
    ///     8
    uint num_faces() const { return _topology.size(2); }

    /// Get number of facets in mesh.
    ///
    /// *Returns*
    ///     integer
    ///         Number of facets.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.num_facets()
    ///     0
    ///     >>> mesh.init(0,1)
    ///     >>> mesh.num_facets()
    ///     16
    uint num_facets() const { return _topology.size(_topology.dim() - 1); }

    /// Get number of cells in mesh.
    ///
    /// *Returns*
    ///     integer
    ///         Number of cells.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.num_cells()
    ///     8
    uint num_cells() const { return _topology.size(_topology.dim()); }

    /// Get number of entities of given topological dimension.
    ///
    /// *Arguments*
    ///     d
    ///         An integer, topological dimension.
    ///
    /// *Returns*
    ///     integer
    ///         Number of entities of topological dimension d.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.init(0,1)
    ///     >>> mesh.num_entities(0)
    ///     9
    ///     >>> mesh.num_entities(1)
    ///     16
    ///     >>> mesh.num_entities(2)
    ///     8
    uint num_entities(uint d) const { return _topology.size(d); }

    /// Get vertex coordinates.
    ///
    /// *Returns*
    ///     An array of doubles
    ///         Coordinates of all vertices.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(1,1)
    ///     >>> mesh.coordinates()
    ///     array([[ 0.,  0.],
    ///            [ 1.,  0.],
    ///            [ 0.,  1.],
    ///            [ 1.,  1.]])
    double* coordinates() { return _geometry.x(); }

    /// Return coordinates of all vertices (const version).
    const double* coordinates() const { return _geometry.x(); }

    /// Get cell connectivity.
    ///
    /// *Returns*
    ///     An array of integers
    ///         Connectivity for all cells.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(1,1)
    ///     >>> mesh.coordinates()
    ///     array([[0, 1, 3],
    ///            [0, 2, 3]])
    const uint* cells() const { return _topology(_topology.dim(), 0)(); }

    /// Get number of entities of given topological dimension.
    ///
    /// *Arguments*
    ///     dim
    ///         An integer, topological dimension.
    ///
    /// *Returns*
    ///     integer
    ///         Number of entities of topological dimension d.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.init(0,1)
    ///     >>> mesh.num_entities(0)
    ///     9
    ///     >>> mesh.num_entities(1)
    ///     16
    ///     >>> mesh.num_entities(2)
    ///     8
    uint size(uint dim) const { return _topology.size(dim); }

    ///  MeshTopology& topology()
    ///
    /// Get topology associated with mesh.
    ///
    /// *Returns*
    ///     :cpp:class:`MeshTopology`
    ///         The topology object associated with the mesh.
    MeshTopology& topology() { return _topology; }

    /// Get mesh topology (const version).
    const MeshTopology& topology() const { return _topology; }

    /// Get mesh geometry.
    ///
    /// *Returns*
    ///     :cpp:class:`MeshGeometry`
    ///         The geometry object associated with the mesh.
    MeshGeometry& geometry() { return _geometry; }

    /// Get mesh geometry (const version).
    const MeshGeometry& geometry() const { return _geometry; }

    /// Get intersection operator.
    ///
    /// *Returns*
    ///     :cpp:class:`IntersectionOperator`
    ///         The intersection operator object associated with the mesh.
    IntersectionOperator& intersection_operator();

    /// Return intersection operator (const version);
    const IntersectionOperator& intersection_operator() const;

    /// Get mesh data.
    ///
    /// *Returns*
    ///     :cpp:class:`MeshData`
    ///         The mesh data object associated with the mesh.
    MeshData& data() { return _data; }

    /// Get mesh data (const version).
    const MeshData& data() const { return _data; }

    /// Get mesh cell type.
    ///
    /// *Returns*
    ///     :cpp:class:`CellType`
    ///         The cell type object associated with the mesh.
    inline CellType& type() { assert(_cell_type); return *_cell_type; }

    /// Get mesh cell type (const version).
    inline const CellType& type() const { assert(_cell_type); return *_cell_type; }

    /// Compute entities of given topological dimension.
    ///
    ///   *Arguments*
    ///       dim
    ///           An integer, topological dimension.
    ///
    ///   *Returns*
    ///       integer
    ///           Number of created entities.
    uint init(uint dim) const;

    /// Compute connectivity between given pair of dimensions.
    ///
    ///   *Arguments*
    ///       d0
    ///           An integer, topological dimension.
    ///
    ///       d1
    ///           An integer, topological dimension.
    void init(uint d0, uint d1) const;

    /// Compute all entities and connectivity.
    void init() const;

    /// Clear all mesh data.
    void clear();

    /// Order all mesh entities.
    ///
    /// .. seealso::
    ///
    ///     UFC documentation (put link here!)
    void order();

    /// Check if mesh is ordered.
    ///
    /// *Returns*
    ///     bool
    ///         Return true iff topology is ordered according to the UFC
    ///         numbering.
    bool ordered() const;

    /// Move coordinates of mesh according to new boundary coordinates.
    ///
    /// *Arguments*
    ///     boundary
    ///         A :cpp:class:`BoundaryMesh` object.
    ///
    ///     method
    ///         A :cpp:class:`ALEType` (enum).
    ///         Method which defines how the coordinates should be
    ///         moved, default is *hermite*.
    void move(BoundaryMesh& boundary, dolfin::ALEType method=hermite);

    /// Move coordinates of mesh according to adjacent mesh with common global
    /// vertices.
    ///
    /// *Arguments*
    ///     mesh
    ///         A :cpp:class:`Mesh` object.
    ///
    ///     method
    ///         A :cpp:class:`ALEType` (enum).
    ///         Method which defines how the coordinates should be
    ///         moved, default is *hermite*.
    void move(Mesh& mesh, dolfin::ALEType method=hermite);

    /// Move coordinates of mesh according to displacement function.
    ///
    /// *Arguments*
    ///     function
    ///         A :cpp:class:`Function` object.
    void move(const Function& displacement);

    /// Smooth internal vertices of mesh by local averaging.
    ///
    /// *Arguments*
    ///     num_iterations
    ///         An integer, number of iterations to perform smoothing,
    ///         default value is 1.
    void smooth(uint num_iterations=1);

    /// Smooth boundary vertices of mesh by local averaging.
    ///
    /// *Arguments*
    ///     num_iterations
    ///         An integer, number of iterations to perform smoothing,
    ///         default value is 1.
    ///
    ///     harmonic_smoothing
    ///         A bool, flag to turn on harmonics smoothing, default
    ///         value is true.
    void smooth_boundary(uint num_iterations=1, bool harmonic_smoothing=true);

    /// Snap boundary vertices of mesh to match given sub domain.
    ///
    /// *Arguments*
    ///     sub_domain
    ///         A :cpp:class:`SubDomain` object.
    ///
    ///     harmonic_smoothing
    ///         A bool, flag to turn on harmonics smoothing, default
    ///         value is true.
    void snap_boundary(const SubDomain& sub_domain, bool harmonic_smoothing=true);

    /// Compute all ids of all cells which are intersected by the
    /// given point.
    ///
    /// *Arguments*
    ///     point
    ///         A :cpp:class:`Point` object.
    ///
    ///     ids_result
    ///         A set of integers.
    ///         The cell ids which are intersected are stored in a set for
    ///         efficiency reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const Point& point, uint_set& ids_result) const;

    /// Compute all ids of all cells which are intersected by any
    /// point in points.
    ///
    /// *Arguments*
    ///     points
    ///         A vector of :cpp:class:`Point` objects.
    ///
    ///     ids_result
    ///         A set of integers.
    ///         The cell ids which are intersected are stored in a set
    ///         for efficiency reasons, to avoid to sort out
    ///         duplicates later on.
    void all_intersected_entities(const std::vector<Point>& points, uint_set& ids_result) const;

    /// Compute all ids of all cells which are intersected by the given
    /// entity.
    ///
    /// *Arguments*
    ///     entity
    ///         A :cpp:class:`MeshEntity` object.
    ///
    ///     ids_result
    ///         A list of integers.
    ///         The ids of the intersected cells are saved in a list.
    ///         This is more efficent than using a set and allows a
    ///         map between the (external) cell and the intersected
    ///         cell of the mesh.
    void all_intersected_entities(const MeshEntity& entity, std::vector<uint>& ids_result) const;

    /// Compute all id of all cells which are intersected by any entity in the
    /// vector entities.
    ///
    /// *Arguments*
    ///     entities
    ///         A vector of :cpp:class:`MeshEntity` objects.
    ///
    ///     ids_result
    ///         A set of integers.
    ///         The cell ids which are intersected are stored in a set for
    ///         efficiency reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const std::vector<MeshEntity>& entities, uint_set& ids_result) const;

    /// Compute all ids of all cells which are intersected by
    /// another_mesh.
    ///
    /// *Arguments*
    ///     another_mesh
    ///         A :cpp:class:`Mesh` object.
    ///
    ///     ids_result
    ///         A set of integers.
    ///         The cell ids which are intersected are stored in a set for
    ///         efficiency reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const Mesh& another_mesh, uint_set& ids_result) const;

    /// Computes only the first id of the entity, which contains the
    /// point.
    ///
    /// *Arguments*
    ///     point
    ///         A :cpp:class:`Point` object.
    ///
    /// *Returns*
    ///     integer
    ///         The first id of the cell, which contains the point,
    ///         returns -1 if no cell is intersected.
    int any_intersected_entity(const Point& point) const;

    /// Computes the point inside the mesh and the corresponding cell
    /// index which are closest to the point query.
    ///
    /// *Arguments*
    ///     point
    ///         A :cpp:class:`Point` object.
    ///
    /// *Returns*
    ///     :cpp:class:`Point`
    ///         The point inside the mesh which is closest to the
    ///         point.
    Point closest_point(const Point& point) const;

    /// Computes the index of the cell in the mesh which is closest to the
    /// point query.
    ///
    /// *Arguments*
    ///     point
    ///         A :cpp:class:`Point` object.
    ///
    /// *Returns*
    ///     integer
    ///         The index of the cell in the mesh which is closest to point.
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

    /// Computes the point inside the mesh and the corresponding cell
    /// index which are closest to the point query.
    ///
    /// *Arguments*
    ///     point
    ///         A :cpp:class:`Point` object.
    ///
    /// *Returns*
    ///     pair <:cpp:class:`Point`, integer>
    ///         The point inside the mesh and the corresponding cell
    ///         index which is closest to the point query.
    std::pair<Point,dolfin::uint> closest_point_and_cell(const Point& point) const;

    /// Compute minimum cell diameter.
    ///
    /// *Returns*
    ///     double
    ///         The minimum cell diameter, the diameter is computed as
    ///         two times the circumradius
    ///         (http://mathworld.wolfram.com).
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.hmin()
    ///     0.70710678118654757
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
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.hmax()
    ///     0.70710678118654757
    double hmax() const;

    /// Informal string representation.
    ///
    /// *Arguments*
    ///     verbose
    ///         A bool, flag to turn on additional output.
    ///
    /// *Returns*
    ///     string
    ///         An informal representation of the mesh.
    ///
    /// *Example*
    ///     .. warning::
    ///
    ///         Not C++ syntax.
    ///
    ///     >>> mesh = dolfin.UnitSquare(2,2)
    ///     >>> mesh.str(False)
    ///     '<Mesh of topological dimension 2 (triangles) with 9 vertices and 8 cells, ordered>'
    std::string str(bool verbose) const;

    /// Define XMLHandler for use in new XML reader/writer
    typedef XMLMesh XMLHandler;

  private:

    // Friends
    friend class MeshEditor;
    friend class TopologyComputation;
    friend class MeshOrdering;
    friend class AdaptiveObjects;
    friend class BinaryFile;

    // Mesh topology
    MeshTopology _topology;

    // Mesh geometry
    MeshGeometry _geometry;

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
