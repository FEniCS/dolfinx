// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2005-05-17
// Last changed: 2006-05-03

#ifndef __AFFINE_MAP_H
#define __AFFINE_MAP_H

#include <dolfin/constants.h>
#include <dolfin/Point.h>

namespace dolfin
{
  class Cell;

  /// This class represents the affine map from the reference element to
  /// the current element.
  ///
  /// The 2D reference element is given by (0,0)-(1,0)-(0,1).
  /// The 3D reference element is given by (0,0,0)-(1,0,0)-(0,1,0)-(0,0,1).
  ///
  /// The dimension d of the map is automatically determined from the
  /// arguments used when calling the map.

  class AffineMap
  {
  public:
    
    /// Constructor
    AffineMap();

    /// Destructor
    ~AffineMap();

    /// Update map for current element
    void update(Cell& cell);

    /// Update map for facet of current element
    void update(Cell& cell, uint facet);

    /// Map given point from the reference element (2D)
    Point operator() (real X, real Y) const;

    /// Map given point from the reference element (3D)
    Point operator() (real X, real Y, real Z) const;

    /// Return cell of current element
    inline Cell& cell() const { return *_cell; }
    
    // Determinant of Jacobian of map
    real det;

    // Scaling factor for mapping
    real scaling;

    // Jacobian of map
    real f00, f01, f02, f10, f11, f12, f20, f21, f22;

    // Inverse of Jacobian of map
    real g00, g01, g02, g10, g11, g12, g20, g21, g22;

  private:

    // Update affine map from reference triangle
    void updateTriangle(Cell& cell);
    
    // Update affine map from reference tetrahedron
    void updateTetrahedron(Cell& cell);

    // Vertices of current cell
    Point p0, p1, p2, p3;

    // Current cell
    Cell* _cell;

  };

}

#endif
