// Copyright (C) 2006-2011 Anders Logg
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
// Modified by Garth N. Wells 2011
//
// First added:  2006-06-02
// Last changed: 2011-11-14

#include "Cell.h"
#include "ParallelData.h"
#include "Point.h"
#include "Facet.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Facet::normal(uint i) const
{
  const uint D = _mesh->topology().dim();
  _mesh->init(D - 1);
  _mesh->init(D - 1, D);
  assert(_mesh->ordered());

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet, i);
}
//-----------------------------------------------------------------------------
Point Facet::normal() const
{
  const uint D = _mesh->topology().dim();
  _mesh->init(D - 1);
  _mesh->init(D - 1, D);
  assert(_mesh->ordered());

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------
bool Facet::exterior() const
{
  if (_mesh->parallel_data().exterior_facet().size() > 0)
    return _mesh->parallel_data().exterior_facet()[this->index()];
  else
    return num_entities(dim() + 1) == 1;
}
//-----------------------------------------------------------------------------
std::pair<const Cell, const Cell>
Facet::adjacent_cells(const MeshFunction<uint>* facet_orientation) const
{
  assert(num_entities(dim() + 1) == 2);

  // Get cell indices
  const uint D = dim() + 1;
  const uint c0 = entities(D)[0];
  const uint c1 = entities(D)[1];

  // Normal ordering
  if (!facet_orientation || (*facet_orientation)[*this] == c0)
    return std::make_pair(Cell(mesh(), c0), Cell(mesh(), c1));

  // Sanity check
  if ((*facet_orientation)[*this] != c1)
  {
    dolfin_error("Facet.cpp",
                 "extract adjacent cells of facet",
                 "Illegal facet orientation specified, cell %d is not a neighbor of facet %d",
         (*facet_orientation)[*this], index());
  }

  // Opposite ordering
  return std::make_pair(Cell(mesh(), c1), Cell(mesh(), c0));
}
//-----------------------------------------------------------------------------
