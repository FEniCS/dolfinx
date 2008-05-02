// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-05-02

#ifndef __ALE_H
#define __ALE_H

#include <dolfin/common/types.h>
#include "MeshFunction.h"
#include "ALEMethod.h"

namespace dolfin
{

  class Mesh;
  class Vertex;

  /// This class provides functionality useful for implementation of
  /// ALE (Arbitrary Lagrangian-Eulerian) methods, in particular
  /// moving the boundary vertices of a mesh and then interpolating
  /// the new coordinates for the interior vertices accordingly.

  class ALE
  {
  public:

    /// Move coordinates of mesh according to new boundary coordinates
    static void move(Mesh& mesh, Mesh& new_boundary,
                     const MeshFunction<uint>& vertex_map,
                     ALEMethod type = lagrange);
    
  private:
    
    // Transfinite meanvalue interpolation
    static void meanValue(real* new_x, uint dim, Mesh& new_boundary,
                          Mesh& mesh, const MeshFunction<uint>& vertex_map,
                          Vertex& vertex);

    // Compute weights for transfinite meanvalue interpolation
    static void computeWeights(real* w, real** u, real* d,
                               uint dim, uint num_vertices);

    // Return sign
    inline static real sgn(real v)
    { return (v < 0.0 ? -1.0 : 1.0); }

    // Return determinant
    inline static real det(real* u, real* v, real* w)
    { return u[0]*(v[1]*w[2] - v[2]*w[1]) - u[1]*(v[0]*w[2] - v[2]*w[0]) + u[2]*(v[0]*w[1] - v[1]*w[0]); }

    // Return next index
    inline static uint next(uint i, uint dim)
    { return (i == dim - 1 ? 0 : i + 1); }

    // Return previous index
    inline static uint previous(uint i, uint dim)
    { return (i == 0 ? dim - 1 : i - 1); }

    // Return distance
    static real dist(const real* x, const real* y, uint dim);

  };

}

#endif
