// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-09
// Last changed: 2006-05-18

#ifndef __MESH_CONNECTIVITY_H
#define __MESH_CONNECTIVITY_H

#include <dolfin/constants.h>
#include <dolfin/Array.h>

namespace dolfin
{

  /// Mesh connectivity stores a sparse data structure of connections
  /// (incidence relations) between mesh entities for a fixed pair of
  /// topological dimensions.

  class MeshConnectivity
  {
  public:

    /// Create empty connectivity
    MeshConnectivity();

    /// Destructor
    ~MeshConnectivity();

    /// Return total number of connections
    inline uint size() const { return _size; }

    /// Return number of connections for given entity
    inline uint size(uint e) const { return offsets[e + 1] - offsets[e]; }

    /// Return array of connections for given entity
    inline uint* connectivity(uint e) { return connections + offsets[e]; }

    /// Clear all data
    void clear();

    /// Set connectivity
    void set(Array<Array<uint> >& connectivity);

    /// Display data
    void disp() const;
    
  private:

    /// Total number of connections
    uint _size;

    /// Number of entities
    uint num_entities;
    
    /// Connections for all entities stored as a contiguous array
    uint* connections;

    /// Offset for first connection for each entity
    uint* offsets;

  };

}

#endif
