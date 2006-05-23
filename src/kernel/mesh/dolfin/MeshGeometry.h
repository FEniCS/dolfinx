// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-05-16

#ifndef __MESH_GEOMETRY_H
#define __MESH_GEOMETRY_H

#include <dolfin/constants.h>

namespace dolfin
{
  
  /// MeshGeometry stores a set of coordinates associated with the vertices
  /// of a mesh.

  class MeshGeometry
  {
  public:

    /// Create empty set of coordinates
    MeshGeometry();

    /// Destructor
    ~MeshGeometry();

    /// Return Euclidean dimension of coordinate system
    inline uint dim() const { return _dim; }

    /// Return number of coordinates
    inline uint size() const { return _size; }
    
    /// Return value of coordinate n in direction d
    inline double& x(uint n, uint d) { return coordinates[d*_size + n]; }

    /// Return value of coordinate n in direction d
    inline double x(uint n, uint d) const { return coordinates[d*_size + n]; }

    /// Clear all data
    void clear();

    /// Initialize coordinate list to given dimension and size
    void init(uint dim, uint size);

    /// Set value of coordinate n in direction d
    void set(uint n, uint d, real x);

    /// Display data
    void disp() const;
    
  private:
    
    // Coordinates for all vertices stored as a contiguous array
    real* coordinates;

    // Euclidean dimension
    uint _dim;
    
    // Number of coordinates
    uint _size;
    
  };

}

#endif
