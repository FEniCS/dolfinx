// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2005-05-17
// Last changed: 2006-12-06

#ifndef __NEW_AFFINE_MAP_H
#define __NEW_AFFINE_MAP_H

#include <dolfin/constants.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/DenseMatrix.h>

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

  class NewAffineMap
  {
  public:
    
    /// Constructor
    NewAffineMap();

    /// Destructor
    ~NewAffineMap();

    /// Update map for current element
    void update(Cell& cell);

    /// Map given point from the reference element (2D)
    Point operator() (real X, real Y) const;

    /// Map given point from the reference element (3D)
    Point operator() (real X, real Y, real Z) const;

    /// Map given point from the reference element
    Point map(real X, real Y, real Z) const;

    /// Map given point from the reference element
    Point mapinv(real X, real Y, real Z) const;

    // Determinant of Jacobian of map
    real det;

    // Jacobian of map
    DenseMatrix B;

    // Inverse of Jacobian of map
    DenseMatrix C;

  private:

    // Update affine map from reference triangle
    void updateTriangle(Cell& cell);
    
    // Update affine map from reference tetrahedron
    void updateTetrahedron(Cell& cell);

    // Vertices of current cell
    Point p0, p1, p2, p3;

  };

}

#endif
