// Copyright (C) 2006 Anders Logg
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
// Modified by Garth N. Wells, 2011.
//
// First added:  2006-06-02
// Last changed: 2011-02-26

#include <dolfin/geometry/Point.h>
#include "Cell.h"
#include "Face.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Face::area() const
{
  dolfin_assert(_mesh);
  dolfin_assert(_mesh->ordered());

  const std::size_t D = _mesh->topology().dim();

  // If the Face is the same topological dimension as cell
  if (D == 2)
  {
    // Get the cell corresponding to this Face
    const Cell cell(*_mesh, this->index());

    // Return the generalized volume (area)
    return cell.volume();

  }
  else
  {

    // Initialize needed connectivity
    _mesh->init(2, D);

    // Get cell to which face belong (first cell when there is more than one)
    const Cell cell(*_mesh, this->entities(D)[0]);

    // Get local index of facet with respect to the cell
    const std::size_t local_facet = cell.index(*this);

    return cell.facet_area(local_facet);
  }
}
//-----------------------------------------------------------------------------
double Face::normal(std::size_t i) const
{
  dolfin_assert(_mesh);
  dolfin_assert(_mesh->ordered());

  const std::size_t tD = _mesh->topology().dim();
  const std::size_t gD = _mesh->geometry().dim();

  // Check for when Cell has the same topological dimension as Face and we are in R^2
  if (tD == 2 && gD == 2)
  {
    dolfin_error("Face.cpp",
                 "compute Face normal",
                 "Don't know how to compute Face normal for a Face in a 2D mesh embedded in R^2.");
  }

  // Check for when Cell has the same topological dimension as Face and we are in R^3
  if (tD == 2 && gD == 3)
  {
    dolfin_not_implemented();
  }

  // Initialize needed connectivity
  _mesh->init(2, tD);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(tD)[0]);

  // Get local index of facet with respect to the cell
  const std::size_t local_facet = cell.index(*this);

  return cell.normal(local_facet, i);
}
//-----------------------------------------------------------------------------
Point Face::normal() const
{
  dolfin_assert(_mesh);
  dolfin_assert(_mesh->ordered());

  const std::size_t tD = _mesh->topology().dim();
  const std::size_t gD = _mesh->geometry().dim();

  // Check for when Cell has the same topological dimension as Face and we are in R^2
  if (tD == 2 && gD == 2)
  {
    dolfin_error("Face.cpp",
                 "compute Face normal",
                 "Don't know how to compute Face normal for a Face in a 2D mesh embedded in R^2.");
  }

  // Check for when Cell has the same topological dimension as Face and we are in R^3
  if (tD == 2 && gD == 3)
  {
    dolfin_not_implemented();
  }

  // Initialize needed connectivity
  _mesh->init(2, tD);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(tD)[0]);

  // Get local index of facet with respect to the cell
  const std::size_t local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------
