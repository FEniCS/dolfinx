// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-05-08
// Last changed: 2007-11-30

#ifndef __MESH_GEOMETRY_H
#define __MESH_GEOMETRY_H

#include "Point.h"
#include <dolfin/common/types.h>

namespace dolfin
{
  
  /// MeshGeometry stores the geometry imposed on a mesh. Currently,
  /// the geometry is represented by the set of coordinates for the
  /// vertices of a mesh, but other representations are possible.
  
  class Mesh;
  class Function;
  class Vector;
  
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
    inline real& x(uint n, uint i) { dolfin_assert(n < _size && i < _dim); return coordinates[n*_dim + i]; }

    /// Return value of coordinate n in direction i
    inline real x(uint n, uint i) const { dolfin_assert(n < _size && i < _dim); return coordinates[n*_dim + i]; }
    
    /// Return array of values for coordinate n
    inline real* x(uint n) { return coordinates + n*_dim; }

    /// Return array of values for coordinate n
    inline const real* x(uint n) const { return coordinates + n*_dim; }

    /// Return array of values for all coordinates
    inline real* x() { return coordinates; }

    /// Return array of values for all coordinates
    inline const real* x() const { return coordinates; }

    /// Return coordinate n as a 3D point value
    Point point(uint n) const;
    
    /// Return pointer to Function for higher order mesh coordinates
    inline Function* mesh_coord_function() { return mesh_coordinates; }
    
    /// Return pointer to boolean affine indicator array
    inline bool* affine_cell_bool() { return affine_cell; }

    /// Clear all data
    void clear();

    /// Initialize coordinate list to given dimension and size
    void init(uint dim, uint size);
    
    /// Initialize the affine indicator array
    void initAffineIndicator(uint num_cells);    
    
    /// set affine indicator at index i
    void setAffineIndicator(uint i, bool value);
    
    /// Set value of coordinate n in direction i
    void set(uint n, uint i, real x);
    
    /// Set higher order mesh coordinates
    void set_mesh_coordinates(Mesh* mesh,  Vector* mesh_coord_vec,
                              const std::string      FE_signature,
                              const std::string  dofmap_signature);

    /// Display data
    void disp() const;

  private:
    
    friend class MPIMeshCommunicator;

    // Euclidean dimension
    uint _dim;
    
    // Number of coordinates
    uint _size;

    // Coordinates for all vertices stored as a contiguous array
    real* coordinates;
    
    // Higher order mesh coordinates (stored as a discrete function)
    Function* mesh_coordinates;
    
    // Boolean indicator for whether a cell is affinely mapped (or not)
    // note: this is used in conjunction with mesh_coordinates
    bool* affine_cell;

  };

}

#endif
