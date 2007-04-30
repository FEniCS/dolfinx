// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman 2007.
// Modified by Magnus Vikstrøm 2007.
//
// First added:  2006-05-08
// Last changed: 2007-04-24

#ifndef __MESH_H
#define __MESH_H

#include <string>
#include <dolfin/constants.h>
#include <dolfin/Variable.h>
#include <dolfin/MeshData.h>

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
    void refine(MeshFunction<bool>& cell_marker, bool refine_boundary = true);

    /// Coarsen mesh uniformly
    void coarsen();

    /// Coarsen mesh according to cells marked for coarsening
    void coarsen(MeshFunction<bool>& cell_marker, bool coarsen_boundary = false);
    
    /// Smooth mesh using Lagrangian mesh smoothing 
    void smooth();

	 /// Partiton mesh into num_part partitions
	 void partition(uint num_part, MeshFunction<uint>& partitions);

    /// Display mesh data
    void disp() const;
    
    /// Output
    friend LogStream& operator<< (LogStream& stream, const Mesh& mesh);
    
  private:

    // Friends
    friend class MeshEditor;

    // Mesh data
    MeshData data;
    
  };

}

#endif
