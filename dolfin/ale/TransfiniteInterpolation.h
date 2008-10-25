// Copyright (C) 2008 Solveig Bruvoll and Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-05-02
// Last changed: 2008-08-12

#ifndef __TRANSFINITE_INTERPOLATION_H
#define __TRANSFINITE_INTERPOLATION_H

#include <dolfin/common/types.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Facet.h>

namespace dolfin
{

  class Mesh;
  class Vertex;

  /// This class implements mesh smoothing by transfinite meanvalue interpolation.

  class TransfiniteInterpolation
  {
  public:

    /// Type of interpolation
    enum InterpolationType {interpolation_lagrange, interpolation_hermite};

    /// Move coordinates of mesh according to new boundary coordinates
    static void move(Mesh& mesh, Mesh& new_boundary,
                     InterpolationType type=interpolation_lagrange);
    
  private:
    
    // Transfinite meanvalue interpolation
    static void meanValue(double* new_x, uint dim, Mesh& new_boundary,
                          Mesh& mesh, const MeshFunction<uint>& vertex_map,
                          const Vertex& vertex, double** ghat, InterpolationType type);

    // Compute weights for transfinite meanvalue interpolation
    static void computeWeights2D(double* w, double** u, double* d,
                                 uint dim, uint num_vertices);

    // Compute weights for transfinite meanvalue interpolation
    static void computeWeights3D(double* w, double** u, double* d,
                                 uint dim, uint num_vertices);

    static void normals(double** dfdn, uint dim, Mesh& new_boundary,
			Mesh& mesh, const MeshFunction<uint>& vertex_map, 
			const MeshFunction<uint>& cell_map);

    static void hermiteFunction(double** ghat, uint dim, Mesh& new_boundary,
				Mesh& mesh, 
				const MeshFunction<uint>& vertex_map, 
				const MeshFunction<uint>& cell_map);

    static void integral(double* new_x, uint dim, Mesh& new_boundary,
                    Mesh& mesh, const MeshFunction<uint>& vertex_map,
                    const Vertex& vertex);


    // Return sign
    inline static double sgn(double v)
    { return (v < 0.0 ? -1.0 : 1.0); }

    // Return determinant
    inline static double det(double* u, double* v, double* w)
    { return u[0]*(v[1]*w[2] - v[2]*w[1]) - u[1]*(v[0]*w[2] - v[2]*w[0]) + u[2]*(v[0]*w[1] - v[1]*w[0]); }

    // Return next index
    inline static uint next(uint i, uint dim)
    { return (i == dim - 1 ? 0 : i + 1); }

    // Return previous index
    inline static uint previous(uint i, uint dim)
    { return (i == 0 ? dim - 1 : i - 1); }

    // Return distance
    static double dist(const double* x, const double* y, uint dim);

    // Return length
    static double length(const double* x, uint dim);
  };

}

#endif
