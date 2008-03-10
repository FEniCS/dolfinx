// Copyright (C) 2006-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2007.
// Modified by Magnus Vikstrøm 2007.
// Modified by Garth N. Wells 2007.
//
// First added:  2006-05-08
// Last changed: 2008-03-10

#ifndef __MESH_H
#define __MESH_H

#include <string>
#include <dolfin/main/constants.h>
#include <dolfin/common/Variable.h>
#include "MeshData.h"

namespace dolfin
{
  
  class MeshTopology;
  class MeshGeometry;
  class CellType;
  template <class T> class MeshFunction;

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
    Mesh(std::string filename);
    
    /// Destructor
    ~Mesh();

    /// Assignment
    const Mesh& operator=(const Mesh& mesh);

    /// Return number of vertices
    inline uint numVertices() const { return data.topology.size(0); }

    /// Return number of edges
    inline uint numEdges() const { return data.topology.size(1); }

    /// Return number of faces
    inline uint numFaces() const { return data.topology.size(2); }

    /// Return number of facets
    inline uint numFacets() const { return data.topology.size(data.topology.dim() - 1); }

    /// Return number of cells
    inline uint numCells() const { return data.topology.size(data.topology.dim()); }

    /// Return coordinates of all vertices
    inline real* coordinates() { return data.geometry.x(); }

    /// Return coordinates of all vertices
    inline const real* coordinates() const { return data.geometry.x(); }

    /// Return connectivity for all cells
    inline uint* cells() { return data.topology(data.topology.dim(), 0)(); }

    /// Return connectivity for all cells
    inline const uint* cells() const { return data.topology(data.topology.dim(), 0)(); }

    /// Return number of entities of given topological dimension
    inline uint size(uint dim) const { return data.topology.size(dim); }
    
    /// Return mesh topology
    inline MeshTopology& topology() { return data.topology; }

    /// Return mesh topology
    inline const MeshTopology& topology() const { return data.topology; }

    /// Return mesh geometry
    inline MeshGeometry& geometry() { return data.geometry; }

    /// Return mesh geometry
    inline const MeshGeometry& geometry() const { return data.geometry; }

    /// Return mesh cell type
    inline CellType& type() { dolfin_assert(data.cell_type); return *data.cell_type; }

    /// Return mesh cell type
    inline const CellType& type() const { dolfin_assert(data.cell_type); return *data.cell_type; }

    /// Compute entities of given topological dimension and return number of entities
    uint init(uint dim);

    /// Compute connectivity between given pair of dimensions
    void init(uint d0, uint d1);

    /// Compute all entities and connectivity
    void init();

    /// Order all mesh entities (not needed if "mesh order entities" is set)
    void order();

    /// Refine mesh uniformly
    void refine();

    /// Refine mesh according to cells marked for refinement
    void refine(MeshFunction<bool>& cell_markers, bool refine_boundary = true);

    /// Coarsen mesh uniformly
    void coarsen();

    /// Coarsen mesh according to cells marked for coarsening
    void coarsen(MeshFunction<bool>& cell_markers, bool coarsen_boundary = false);
    
    /// Smooth mesh using Lagrangian mesh smoothing 
    void smooth();
    
    /// Partition mesh into num_processes partitions
    void partition(MeshFunction<uint>& partitions);

    /// Partition mesh into num_partitions partitions
    void partition(MeshFunction<uint>& partitions, uint num_partitions);

    /// Display mesh data
    void disp() const;
    
    /// Return a short desriptive string
    std::string str() const;

    /// Output
    friend LogStream& operator<< (LogStream& stream, const Mesh& mesh);
    
  private:

    // Friends
    friend class MeshEditor;
    friend class MPIMeshCommunicator;

    // Mesh data
    MeshData data;
    
  };

}

#endif
