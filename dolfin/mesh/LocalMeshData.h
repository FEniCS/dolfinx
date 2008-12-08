// Copyright (C) 2008 Skavhaug.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-28
// Last changed: 2008-12-02
//
// Modified by Anders Logg, 2008.

#ifndef __LOCALMESHDATA_H
#define __LOCALMESHDATA_H

#include <vector>
#include <dolfin/common/types.h>

namespace dolfin
{

  class XMLLocalMeshData;
  
  /// This class stores mesh data on a local processor corresponding
  /// to a portion of a (larger) global mesh.
  ///
  /// Note that the data stored in this class does typically not
  /// correspond to a topologically connected mesh; it merely stores a
  /// list of vertex coordinates, a list of cell-vertex mappings and a
  /// list of global vertex numbers for the locally stored vertices.
  ///
  /// It is typically used for parsing meshes in parallel from mesh
  /// XML files. After local mesh data has been parsed on each
  /// processor, a subsequent repartitioning takes place: first a
  /// geometric partitioning of the vertices followed by a
  /// redistribution of vertex and cell data, and then a topological
  /// partitioning again followed by redistribution of vertex and cell
  /// data, at that point corresponding to topologically connected
  /// meshes instead of local mesh data.

  class LocalMeshData
  {
  public:
    
    /// Constructor
    LocalMeshData();
    
    /// Destructor
    ~LocalMeshData();

    /// Clear all data
    void clear();

    /// Read-only access to vertex coordinates
    const std::vector<std::vector<double> >& vertex_coordinates() const
    { return _vertex_coordinates; }
    
    /// Read-only access to vertex indices
    const std::vector<uint>& vertex_indices() const
    { return _vertex_indices; }
    
    /// Read-only access to cell vertices
    const std::vector<std::vector<uint> >& cell_vertices() const
    { return _cell_vertices; }

  private:
    
    /// Coordinates for all vertices stored on local processor
    std::vector<std::vector<double> > _vertex_coordinates;

    /// Global vertex indices for all vertices stored on local processor
    std::vector<uint> _vertex_indices;

    /// Global vertex indices for all cells stored on local processor
    std::vector<std::vector<uint> > _cell_vertices;
    
    // Friends
    friend class XMLLocalMeshData;
    
  };
  
}

#endif
