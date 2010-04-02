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
// Last changed: 2010-03-03

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
  /// Both the representation and the interface are dimension-independent,
  /// but a concrete interface is also provided for standard named mesh
  /// entities:
  ///
  ///     Entity  Dimension  Codimension
  ///
  ///     Vertex      0           -
  ///     Edge        1           -
  ///     Face        2           -
  ///
  ///     Facet       -           1
  ///     Cell        -           0
  ///
  /// When working with mesh iterators, all entities and connectivity
  /// are precomputed automatically the first time an iterator is
  /// created over any given topological dimension or connectivity.
  ///
  /// Note that for efficiency, only entities of dimension zero
  /// (vertices) and entities of the maximal dimension (cells) exist
  /// when creating a Mesh. Other entities must be explicitly created
  /// by calling init(). For example, all edges in a mesh may be created
  /// by a call to mesh.init(1). Similarly, connectivities such as
  /// all edges connected to a given vertex must also be explicitly
  /// created (in this case by a call to mesh.init(0, 1)).

  class Mesh : public Variable
  {
  public:

    /// Create empty mesh
    Mesh();

    /// Copy constructor
    Mesh(const Mesh& mesh);

    /// Create mesh from data file
    explicit Mesh(std::string filename);

    /// Destructor
    ~Mesh();

    /// Assignment
    const Mesh& operator=(const Mesh& mesh);

    /// Return number of vertices
    uint num_vertices() const { return _topology.size(0); }

    /// Return number of edges
    uint num_edges() const { return _topology.size(1); }

    /// Return number of faces
    uint num_faces() const { return _topology.size(2); }

    /// Return number of facets
    uint num_facets() const { return _topology.size(_topology.dim() - 1); }

    /// Return number of cells
    uint num_cells() const { return _topology.size(_topology.dim()); }

    /// Return number of entities of dimension d
    uint num_entities(uint d) const { return _topology.size(d); }

    /// Return coordinates of all vertices
    double* coordinates() { return _geometry.x(); }

    /// Return coordinates of all vertices
    const double* coordinates() const { return _geometry.x(); }

    /// Return connectivity for all cells
    const uint* cells() const { return _topology(_topology.dim(), 0)(); }

    /// Return number of entities of given topological dimension
    uint size(uint dim) const { return _topology.size(dim); }

    /// Return mesh topology (non-const version)
    MeshTopology& topology() { return _topology; }

    /// Return mesh topology (const version)
    const MeshTopology& topology() const { return _topology; }

    /// Return mesh geometry (non-const version)
    MeshGeometry& geometry() { return _geometry; }

    /// Return mesh geometry (const version)
    const MeshGeometry& geometry() const { return _geometry; }

    ///Return intersectionoperator (const version);
    const IntersectionOperator& intersection_operator() const;

    ///Return intersectionoperator (non-const version);
    IntersectionOperator& intersection_operator();

    /// Return mesh data (non-const version)
    MeshData& data() { return _data; }

    /// Return mesh data (const version)
    const MeshData& data() const { return _data; }

    /// Return mesh cell type
    inline CellType& type() { assert(_cell_type); return *_cell_type; }

    /// Return mesh cell type
    inline const CellType& type() const { assert(_cell_type); return *_cell_type; }

    /// Compute entities of given topological dimension and return number of entities
    uint init(uint dim) const;

    /// Compute connectivity between given pair of dimensions
    void init(uint d0, uint d1) const;

    /// Compute all entities and connectivity
    void init() const;

    /// Clear all mesh data
    void clear();

    /// Order all mesh entities (not needed if "mesh order entities" is set)
    void order();

    /// Return true iff topology is ordered according to the UFC numbering
    bool ordered() const;

    /// Move coordinates of mesh according to new boundary coordinates
    void move(BoundaryMesh& boundary, dolfin::ALEType method=hermite);

    /// Move coordinates of mesh according to adjacent mesh with common global vertices
    void move(Mesh& mesh, dolfin::ALEType method=hermite);

    /// Move coordinates of mesh according to displacement function
    void move(const Function& displacement);

    /// Smooth internal vertices of mesh by local averaging
    void smooth(uint num_iterations=1);

    /// Smooth boundary vertices of mesh by local averaging
    void smooth_boundary(uint num_iterations=1, bool harmonic_smoothing=true);

    /// Snap boundary vertices of mesh to match given sub domain
    void snap_boundary(const SubDomain& sub_domain, bool harmonic_smoothing=true);

    ///Compute all id of all cells which are intersects by a \em point.
    ///\param[out] ids_result The ids of the intersected entities are saved in a set for efficienty
    ///reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const Point & point, uint_set & ids_result) const;

    ///Compute all id of all cells which are intersects any point in \em points.
    ///\param[out] ids_result The ids of the intersected entities are saved in a set for efficienty
    ///reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const std::vector<Point> & points, uint_set & ids_result) const;

    ///Compute all id of all cells which are intersects by a \em entity.
    ///\param[out] ids_result The ids of the intersected entities are saved in a vector.
    ///This allows is more efficent than using a set and allows a map between
    //the (external) cell and the intersected cell of the mesh. If you
    //are only interested in intersection with a list of cells without caring about which
    //cell what intersected by which one, use
    // void IntersectionOperator::all_intersected_entities(const std::vector<Cell> &, uint_set &) const;
    void all_intersected_entities(const MeshEntity & entity, std::vector<uint> & ids_result) const;

    ///Compute all id of all cells which are intersects by any of the entities in \em entities. This
    ///\param[out] ids_result The ids of the intersected set are saved in a set for efficienty
    ///reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const std::vector<MeshEntity> & entities, uint_set & ids_result) const;

    ///Compute all id of all cells which are intersects by the given mesh \em another_mesh;
    ///\param[out] ids_result The ids of the intersected entities are saved in a set for efficienty
    ///reasons, to avoid to sort out duplicates later on.
    void all_intersected_entities(const Mesh & another_mesh, uint_set & ids_result) const;

    ///Computes only the first id  of the entity, which contains the point. Returns -1 if no cell is intersected.
    ///@internal @remark This makes the function evaluation significantly faster.
    int any_intersected_entity(const Point & point) const;

    ///Computes the point inside the mesh which are closest to the point query.
    Point closest_point(const Point & point) const;

    ///Computes the index of the cell in the mesh
    ///which are closest to the point query.
    dolfin::uint closest_cell(const Point & point) const;

    ///Computes the point inside the mesh and the corresponding cell index
    ///which are closest to the point query.
    std::pair<Point,dolfin::uint> closest_point_and_cell(const Point & point) const;

    /// Compute minimum cell diameter
    double hmin() const;

    /// Compute maximum cell diameter
    double hmax() const;

    /// Return informal string representation (pretty-print)
    std::string str(bool verbose) const;

    /// Define XMLHandler for use in new XML reader/writer
    typedef XMLMesh XMLHandler;

  private:

    // Friends
    friend class MeshEditor;
    friend class TopologyComputation;
    friend class MeshOrdering;
    friend class AdaptiveObjects;

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
