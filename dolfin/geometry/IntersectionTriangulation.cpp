// Copyright (C) 2013 Anders Logg
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
// First added:  2014-02-03
// Last changed: 2014-02-03

#include <dolfin/mesh/Cell.h>
#include "IntersectionTriangulation.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
std::vector<double>
triangulate_intersection_interval_interval(const Cell& T0,
                                           const Cell& T1)
{
  dolfin_assert(T0.mesh().topology().dim() == 1);
  dolfin_assert(T1.mesh().topology().dim() == 1);

  dolfin_not_implemented();
  std::vector<double> triangulation;
  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
triangulate_intersection_triangulate_triangulate(const Cell& T0,
                                                 const Cell& T1)
{
  dolfin_assert(T0.mesh().topology().dim() == 2);
  dolfin_assert(T1.mesh().topology().dim() == 2);

  dolfin_not_implemented();
  std::vector<double> triangulation;
  return triangulation;
}
//-----------------------------------------------------------------------------
std::vector<double>
triangulate_intersection_tetrahedron_tetrahedron(const Cell& T0,
                                                 const Cell& T1)
{
  dolfin_assert(T0.mesh().topology().dim() == 3);
  dolfin_assert(T1.mesh().topology().dim() == 3);

  dolfin_not_implemented();
  std::vector<double> triangulation;
  return triangulation;
}
//-----------------------------------------------------------------------------
