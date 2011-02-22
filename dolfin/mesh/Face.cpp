// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2011.
//
// First added:  2006-06-02
// Last changed: 2011-02-22

#include "Cell.h"
#include "Point.h"
#include "Face.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Face::area() const
{
  // Get mesh geometry
  const MeshGeometry& geometry = mesh().geometry();

  // Get the coordinates of the three vertices
  const uint* vertices = entities(0);
  const double* x0 = geometry.x(vertices[0]);
  const double* x1 = geometry.x(vertices[1]);
  const double* x2 = geometry.x(vertices[2]);

  // Compute area of triangle embedded in R^3
  double v0 = (x0[1]*x1[2] + x0[2]*x2[1] + x1[1]*x2[2]) - (x2[1]*x1[2] + x2[2]*x0[1] + x1[1]*x0[2]);
  double v1 = (x0[2]*x1[0] + x0[0]*x2[2] + x1[2]*x2[0]) - (x2[2]*x1[0] + x2[0]*x0[2] + x1[2]*x0[0]);
  double v2 = (x0[0]*x1[1] + x0[1]*x2[0] + x1[0]*x2[1]) - (x2[0]*x1[1] + x2[1]*x0[0] + x1[0]*x0[1]);

  // Formula for area from http://mathworld.wolfram.com
  return 0.5*sqrt(v0*v0 + v1*v1 + v2*v2);
}
//-----------------------------------------------------------------------------
double Face::normal(uint i) const
{
  _mesh->init(2);
  _mesh->init(2, 3);
  assert(_mesh->ordered());

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(3)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet, i);
}
//-----------------------------------------------------------------------------
Point Face::normal() const
{
  _mesh->init(2);
  _mesh->init(2, 3);
  assert(_mesh->ordered());

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(3)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------

