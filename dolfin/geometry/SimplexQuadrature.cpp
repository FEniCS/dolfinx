// Copyright (C) 2014 Anders Logg
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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2014-02-24
// Last changed: 2014-03-06

#include <dolfin/log/log.h>
#include <dolfin/mesh/Cell.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/MeshGeometry.h>
#include "SimplexQuadrature.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double> >
SimplexQuadrature::compute_quadrature_rule(const Cell& cell, std::size_t order)
{
  // Extract dimensions
  const std::size_t tdim = cell.mesh().topology().dim();
  const std::size_t gdim = cell.mesh().geometry().dim();

  // Get vertex coordinates
  std::vector<double> coordinates;
  cell.get_vertex_coordinates(coordinates);

  // Call function to compute quadrature rule
  return compute_quadrature_rule(&coordinates[0], tdim, gdim, order);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double> >
SimplexQuadrature::compute_quadrature_rule(const double* coordinates,
                                           std::size_t tdim,
                                           std::size_t gdim,
                                           std::size_t order)
{
  switch (tdim)
  {
  case 1:
    return compute_quadrature_rule_interval(coordinates, gdim, order);
    break;
  case 2:
    return compute_quadrature_rule_triangle(coordinates, gdim, order);
    break;
  case 3:
    return compute_quadrature_rule_tetrahedron(coordinates, gdim, order);
    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for simplex",
                 "Only implemented for topological dimension 1, 2, 3");
  };

  std::pair<std::vector<double>, std::vector<double> > quadrature_rule;
  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double> >
SimplexQuadrature::compute_quadrature_rule_interval(const double* coordinates,
                                                    std::size_t gdim,
                                                    std::size_t order)
{
  std::pair<std::vector<double>, std::vector<double> > quadrature_rule;

  // Weights and points in local coordinates on [-1, 1]
  std::vector<double> w, p;

  switch (order)
  {
  case 1: {
    // Assign weight 2, point 0
    w.assign(1, 2.);
    p.assign(1, 0.);

    break;
  }
  case 2: {
    // Assign weights 1.
    w.assign(2, 1.);

    // Assign points corresponding to -1/sqrt(3) and 1/sqrt(3)
    p.resize(2);
    p[0] = -1./std::sqrt(3);
    p[1] = 1./std::sqrt(3);

    break;
  }
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for interval",
                 "Not implemented for order ",order);
  }


  // Find the determinant of the Jacobian (inspired by ufc_geometry.h)
  double det;

  switch (gdim)
  {
  case 1:
    det = coordinates[1] - coordinates[0];
    break;

  case 2:
    {
      const double J[] = {coordinates[2] - coordinates[0],
                          coordinates[3] - coordinates[1]};
      const double det2 = J[0]*J[0] + J[1]*J[1];
      det = std::sqrt(det2);
      break;
    }
  case 3:
    {
      const double J[] = {coordinates[3] - coordinates[0],
                          coordinates[4] - coordinates[1],
                          coordinates[5] - coordinates[2]};
      const double det2 = J[0]*J[0] + J[1]*J[1] + J[2]*J[2];
      det = std::sqrt(det2);
      break;
    }
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for interval",
                 "Not implemented for dimension ", gdim);
  }

  // Store weights
  quadrature_rule.first.assign(w.size(), det);
  for (std::size_t i = 0; i < w.size(); ++i)
    quadrature_rule.first[i] *= w[i];

  // Map (local) quadrature points
  quadrature_rule.second.resize(gdim*p.size());

  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t d = 0; d < gdim; ++d)
      quadrature_rule.second[d+i*gdim] = 0.5*(coordinates[d]*(1-p[i])
                                              + coordinates[gdim+d]*(1+p[i]));

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double> >
SimplexQuadrature::compute_quadrature_rule_triangle(const double* coordinates,
                                                    std::size_t gdim,
                                                    std::size_t order)
{
  std::pair<std::vector<double>, std::vector<double> > quadrature_rule;

  // Weights and points in local coordinates on triangle [0,0], [1,0]
  // and [0,1]
  std::vector<double> w;
  std::vector<std::vector<double> > p;

  switch (order)
  {
  case 1:
    // Assign weight 1 and midpoint
    w.assign(1, 1.);
    p.assign(1, std::vector<double>(3, 1./3));

    break;
  case 2:
    // Assign weight 1/3
    w.assign(2, 1./3);

    // Assign points corresponding to 2/3, 1/6, 1/6
    p.assign(3, std::vector<double>(3, 1./6));
    p[0][0] = p[1][1] = p[2][2] = 2./3;

    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "Not implemented for order ",order);
  }

  // Find the determinant of the Jacobian (inspired by ufc_geometry.h)
  double det;

  switch (gdim)
  {
  case 2:
    {
      const double J[] = {coordinates[2] - coordinates[0],
                          coordinates[4] - coordinates[0],
                          coordinates[3] - coordinates[1],
                          coordinates[5] - coordinates[1]};
      det = J[0]*J[3] - J[1]*J[2];
      break;
    }
  case 3:
    {
      const double J[]={coordinates[3] - coordinates[0],
                        coordinates[6] - coordinates[0],
                        coordinates[4] - coordinates[1],
                        coordinates[7] - coordinates[1],
                        coordinates[5] - coordinates[2],
                        coordinates[8] - coordinates[2]};
      const double d_0 = J[2]*J[5] - J[4]*J[3];
      const double d_1 = J[4]*J[1] - J[0]*J[5];
      const double d_2 = J[0]*J[3] - J[2]*J[1];

      // const double c[] = {J[0]*J[0] + J[2]*J[2] + J[4]*J[4],
      //                     J[1]*J[1] + J[3]*J[3] + J[5]*J[5],
      //                     J[0]*J[1] + J[2]*J[3] + J[4]*J[5]};
      // const double den = c[0]*c[1] - c[2]*c[2];
      const double det2 = d_0*d_0 + d_1*d_1 + d_2*d_2;
      det = std::sqrt(det2);

      break;
    }
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "Not implemented for dimension ", gdim);
  }


  // Store weights
  quadrature_rule.first.assign(w.size(),det);
  for (std::size_t i = 0; i < w.size(); ++i)
    quadrature_rule.first[i] *= w[i];

  // Store points
  quadrature_rule.second.resize(gdim*p.size());

  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t d = 0; d < gdim; ++d)
      quadrature_rule.second[d+i*gdim] = p[i][0]*coordinates[d]
        + p[i][1]*coordinates[gdim+d]
        + p[i][2]*coordinates[2*gdim+d];

  return quadrature_rule;

  // FIXME: Temporary implementation just so we have something to work with

  // // Compute area of triangle
  // const double* x0 = coordinates;
  // const double* x1 = coordinates + 2;
  // const double* x2 = coordinates + 4;
  // const double A = 0.5*std::abs((x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1]) -
  //                               (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]));

  // // Quadrature weights
  // std::vector<double> w;
  // w.push_back(A / 3);
  // w.push_back(A / 3);
  // w.push_back(A / 3);

  // // Quadrature points
  // std::vector<double> x;
  // x.push_back(x0[0]); x.push_back(x0[1]);
  // x.push_back(x1[0]); x.push_back(x1[1]);
  // x.push_back(x2[0]); x.push_back(x2[1]);

  // return std::make_pair(w, x);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double> >
SimplexQuadrature::compute_quadrature_rule_tetrahedron(const double* coordinates,
                                                       std::size_t gdim,
                                                       std::size_t order)
{
  std::pair<std::vector<double>, std::vector<double> > quadrature_rule;

  // Weights and points in local coordinates on tetrahedron [0,0,0],
  // [1,0,0], [0,1,0] and [0,0,1]
  std::vector<double> w;
  std::vector<std::vector<double> > p;

  switch (order)
  {
  case 1:
    // Assign weight 1 and midpoint
    w.assign(1, 1.);
    p.assign(1, std::vector<double>(4, 0.25));

    break;
  case 2:
    // Assign weight 0.25
    w.assign(4, 0.25);

    // Assign points corresponding to 0.585410196624969,
    // 0.138196601125011, 0.138196601125011, 0.138196601125011
    p.assign(4, std::vector<double>(4, 0.138196601125011));
    p[0][0] = p[1][1] = p[2][2] = p[3][3] = 0.585410196624969;

    break;
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for triangle",
                 "Not implemented for order ",order);
  }

  double det;

  switch (gdim)
  {
  case 3:
    {
      const double J[]={coordinates[3]  - coordinates[0],
                        coordinates[6]  - coordinates[0],
                        coordinates[9]  - coordinates[0],
                        coordinates[4]  - coordinates[1],
                        coordinates[7]  - coordinates[1],
                        coordinates[10] - coordinates[1],
                        coordinates[5]  - coordinates[2],
                        coordinates[8]  - coordinates[2],
                        coordinates[11] - coordinates[2]};
      double d[9];
      d[0*3 + 0] = J[4]*J[8] - J[5]*J[7];
      // d[0*3 + 1] = J[5]*J[6] - J[3]*J[8];
      // d[0*3 + 2] = J[3]*J[7] - J[4]*J[6];
      d[1*3 + 0] = J[2]*J[7] - J[1]*J[8];
      // d[1*3 + 1] = J[0]*J[8] - J[2]*J[6];
      // d[1*3 + 2] = J[1]*J[6] - J[0]*J[7];
      d[2*3 + 0] = J[1]*J[5] - J[2]*J[4];
      // d[2*3 + 1] = J[2]*J[3] - J[0]*J[5];
      // d[2*3 + 2] = J[0]*J[4] - J[1]*J[3];

      det = J[0]*d[0*3 + 0] + J[3]*d[1*3 + 0] + J[6]*d[2*3 + 0];
    }
  default:
    dolfin_error("SimplexQuadrature.cpp",
                 "compute quadrature rule for tetrahedron",
                 "Not implemented for dimension ", gdim);
  }

  // Store weights
  quadrature_rule.first.assign(w.size(),det);
  for (std::size_t i = 0; i < w.size(); ++i)
    quadrature_rule.first[i] *= w[i];

  // Store points
  quadrature_rule.second.resize(gdim*p.size());

  for (std::size_t i = 0; i < p.size(); ++i)
    for (std::size_t d = 0; d < gdim; ++d)
      quadrature_rule.second[d+i*gdim] = p[i][0]*coordinates[d]
        + p[i][1]*coordinates[gdim+d]
        + p[i][2]*coordinates[2*gdim+d]
        + p[i][3]*coordinates[3*gdim+d];

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
