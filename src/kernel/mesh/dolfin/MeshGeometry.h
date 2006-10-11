// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-05-08
// Last changed: 2006-10-11

#ifndef __MESH_GEOMETRY_H
#define __MESH_GEOMETRY_H

#include <dolfin/NewPoint.h>
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

    /// Copy constructor
    MeshGeometry(const MeshGeometry& geometry);

    /// Destructor
    ~MeshGeometry();

    /// Assignment
    const MeshGeometry& operator= (const MeshGeometry& geometry);

    /// Return Euclidean dimension of coordinate system
    inline uint dim() const { return _dim; }

    /// Return number of coordinates
    inline uint size() const { return _size; }
    
    /// Return value of coordinate n in direction i
    inline real& x(uint n, uint i) { dolfin_assert(n < _size && i < _dim); return coordinates[i*_size + n]; }

    /// Return value of coordinate n in direction i
    inline real x(uint n, uint i) const { dolfin_assert(n < _size && i < _dim); return coordinates[i*_size + n]; }

    /// Return coordinates as one contiguous array
    inline real* x() { return coordinates; }

    /// Return coordinates as one contiguous array
    inline const real* x() const { return coordinates; }

    /// Return coordinate n as a 3D point value
    NewPoint point(uint n) const;
    
    /// Clear all data
    void clear();

    /// Initialize coordinate list to given dimension and size
    void init(uint dim, uint size);

    /// Set value of coordinate n in direction i
    void set(uint n, uint i, real x);

    /// Display data
    void disp() const;
    
  private:
    
    // Euclidean dimension
    uint _dim;
    
    // Number of coordinates
    uint _size;

    // Coordinates for all vertices stored as a contiguous array
    real* coordinates;
    
  };

}

#endif
