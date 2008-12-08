// Copyright (C) 2006 Skavhaug.
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
  
  class LocalMeshData
  {
  public:
    
    /// Constructor
    LocalMeshData();
    
    /// Destructor
    ~LocalMeshData();

    /// Clear all data
    void clear();

    /// Read only access do vertex coordinates
    const std::vector<std::vector<double> >& vertex_coordinates() const
    { return _vertex_coordinates; }
    
    /// Read only access do vertex indices
    const std::vector<uint>& vertex_indices() const
    { return _vertex_indices; }
    
    /// Read only access do cell vertices
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
