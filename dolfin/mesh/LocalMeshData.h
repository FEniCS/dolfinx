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
    
  private:
    
    /// Coordinates for all vertices stored on local processor
    std::vector<std::vector<double> > vertex_coordinates;

    /// Global vertex indices for all vertices stored on local processor
    std::vector<uint> vertex_indices;

    /// Global vertex indices for all cells stored on local processor
    std::vector<std::vector<uint> > cell_vertices;
    
    // Friends
    friend class XMLLocalMeshData;
    
  };
  
}

#endif
