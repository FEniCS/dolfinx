// Copyright (C) 2008-2009 Solveig Bruvoll and Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2008-05-02
// Last changed: 2009-01-12

#ifndef __TRANSFINITE_INTERPOLATION_H
#define __TRANSFINITE_INTERPOLATION_H

#include <dolfin/common/types.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Facet.h>

namespace dolfin
{

  class BoundaryMesh;
  class Mesh;
  class Vertex;

  /// This class implements mesh smoothing by transfinite meanvalue interpolation.

  class TransfiniteInterpolation
  {
  public:

    /// Type of interpolation
    enum InterpolationType {interpolation_lagrange, interpolation_hermite};

    /// Move coordinates of mesh according to new boundary coordinates
    static void move(Mesh& mesh,const BoundaryMesh& new_boundary,
                     InterpolationType type);//=interpolation_lagrange);

  private:

    // Transfinite meanvalue interpolation
    static void mean_value(Point& new_x, uint dim,
                           const BoundaryMesh& new_boundary,
                           Mesh& mesh,
                           const MeshFunction<uint>& vertex_map,
                           const Vertex& vertex,
                           const std::vector<Point>& ghat,
                           InterpolationType type);

    // Compute weights for transfinite meanvalue interpolation
    static void computeWeights2D(std::vector<double>& w,
                                 const std::vector<Point>& u,
                                 const std::vector<double>& d,
                                 uint num_vertices);

    // Compute weights for transfinite meanvalue interpolation
    static void computeWeights3D(std::vector<double>& w,
                                 const std::vector<Point>& u,
                                 const std::vector<double>& d,
                                 uint dim, uint num_vertices);

    static void normals(double** dfdn, uint dim, const Mesh& new_boundary,
                        Mesh& mesh, const MeshFunction<uint>& vertex_map,
                        const MeshFunction<uint>& cell_map);

    static void hermite_function(std::vector<Point>& ghat, uint dim,
                                 const BoundaryMesh& new_boundary, Mesh& mesh,
                                 const MeshFunction<uint>& vertex_map,
                                 const MeshFunction<uint>& cell_map);

    static void integral(Point& new_x, uint dim,
                         const BoundaryMesh& new_boundary,
                          Mesh& mesh, const MeshFunction<uint>& vertex_map,
                          const Vertex& vertex);


    // Return sign
    static double sgn(double v)
    { return (v < 0.0 ? -1.0 : 1.0); }

    // Return determinant 2D
    static double det(const Point& u, const Point& v)
    { return (u[0]*v[1] - u[1]*v[0]); }

    // Return determinant 3D
    static double det(const Point& u, const Point& v, const Point& w)
    { return (u[0]*(v[1]*w[2] - v[2]*w[1]) - u[1]*(v[0]*w[2] - v[2]*w[0]) + u[2]*(v[0]*w[1] - v[1]*w[0])); }

    // Return next index
    static uint next(uint i, uint dim)
    { return (i == dim - 1 ? 0 : i + 1); }

    // Return previous index
    static uint previous(uint i, uint dim)
    { return (i == 0 ? dim - 1 : i - 1); }

    // Return distance
    static double dist(const Point& x, const Point& y, uint dim);

    // Return length
    static double length(const double* x, uint dim);
  };

}

#endif
