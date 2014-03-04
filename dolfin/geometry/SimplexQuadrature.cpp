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
// Last changed: 2014-03-04

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
  // FIXME: Use function cell.get_vertex_coordinates which does exactly
  // the same thing as the code here.

  // Extract dimensions
  const std::size_t tdim = cell.mesh().topology().dim();
  const std::size_t gdim = cell.mesh().geometry().dim();

  // Extract vertex coordinates
  const MeshGeometry& geometry = cell.mesh().geometry();
  const unsigned int* vertices = cell.entities(0);
  const std::size_t num_vertices = cell.num_entities(0);
  std::vector<double> coordinates(num_vertices*gdim);
  for (unsigned int i = 0; i < num_vertices; i++)
  {
    const double* x = geometry.x(vertices[i]);
    for (unsigned int j = 0; j < gdim; j++)
      coordinates[i*gdim + j] = x[j];
  }

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

  // FIXME: Not implemented

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double> >
SimplexQuadrature::compute_quadrature_rule_triangle(const double* coordinates,
                                                    std::size_t gdim,
                                                    std::size_t order)
{
  std::pair<std::vector<double>, std::vector<double> > quadrature_rule;

  // FIXME: Temporary implementation just so we have something to work with

  // Compute area of triangle
  const double* x0 = coordinates;
  const double* x1 = coordinates + 2;
  const double* x2 = coordinates + 4;
  const double A = 0.5*std::abs((x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1]) -
                                (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]));

  // Quadrature weights
  std::vector<double> w;
  w.push_back(A / 3);
  w.push_back(A / 3);
  w.push_back(A / 3);

  // Quadrature points
  std::vector<double> x;
  x.push_back(x0[0]); x.push_back(x0[1]);
  x.push_back(x1[0]); x.push_back(x1[1]);
  x.push_back(x2[0]); x.push_back(x2[1]);

  return std::make_pair(w, x);
}
//-----------------------------------------------------------------------------
std::pair<std::vector<double>, std::vector<double> >
SimplexQuadrature::compute_quadrature_rule_tetrahedron(const double* coordinates,
                                                       std::size_t gdim,
                                                       std::size_t order)
{
  std::pair<std::vector<double>, std::vector<double> > quadrature_rule;

  // FIXME: Not implemented

  return quadrature_rule;
}
//-----------------------------------------------------------------------------
