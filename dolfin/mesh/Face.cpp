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

#include "Cell.h"
#include "Point.h"
#include "Face.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Face::area() const
{
  assert(_mesh);
  assert(_mesh->ordered());

  // Initialize needed connectivity
  const uint D = _mesh->topology().dim();
  _mesh->init(2, D);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.facet_area(local_facet);
}
//-----------------------------------------------------------------------------
double Face::normal(uint i) const
{
  assert(_mesh);
  assert(_mesh->ordered());

  // Initialize needed connectivity
  const uint D = _mesh->topology().dim();
  _mesh->init(2, D);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet, i);
}
//-----------------------------------------------------------------------------
Point Face::normal() const
{
  assert(_mesh);
  assert(_mesh->ordered());

  // Initialize needed connectivity
  const uint D = _mesh->topology().dim();
  _mesh->init(2, D);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------
