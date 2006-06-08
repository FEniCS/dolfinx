// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-06-08

#ifndef __NEW_MESH_H
#define __NEW_MESH_H

#include <string>
#include <dolfin/constants.h>
#include <dolfin/NewMeshData.h>

namespace dolfin
{

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
  
  class NewMesh
  {
  public:
    
    /// Create empty mesh
    NewMesh();

    /// Copy constructor
    NewMesh(const NewMesh& mesh);

    /// Create mesh from given file
    NewMesh(std::string filename);
    
    /// Destructor
    ~NewMesh();

    /// Return topological dimension
    inline uint dim() const { return data.topology.dim(); }
    
    /// Return number of entities of given topological dimension
    inline uint size(uint dim) const { return data.topology.size(dim); }

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

    /// Compute entities of given topological dimension (automated for iterators)
    void init(uint dim);

    /// Compute connectivity between given pair of dimensions (automated for iterators)
    void init(uint d0, uint d1);

    /// Compute all entities and connectivity (automated for iterators)
    void init();

    /// Display mesh
    void disp() const;
    
    /// Output
    friend LogStream& operator<< (LogStream& stream, const NewMesh& mesh);

  private:
 
    /// Friends
    friend class MeshEditor;
    friend class MeshAlgorithms;
    friend class MeshEntity;
    friend class MeshEntityIterator;

    /// Mesh data
    NewMeshData data;
    
  };

}

#endif
