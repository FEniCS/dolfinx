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
// Modified by Kristian B. Oelgaard, 2007, 2008.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008, 2009.
//
// First added:  2008-07-17
// Last changed: 2011-11-16

#include <dolfin/log/log.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include "SpecialFunctions.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshCoordinates::MeshCoordinates(std::shared_ptr<const Mesh> mesh)
  : Expression(mesh->geometry().dim()), _mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshCoordinates::eval(Array<double>& values, const Array<double>& x,
                           const ufc::cell& cell) const
{
  dolfin_assert(_mesh);
  dolfin_assert(cell.geometric_dimension == _mesh->geometry().dim());
  dolfin_assert(x.size() == _mesh->geometry().dim());

  for (std::size_t i = 0; i < cell.geometric_dimension; ++i)
    values[i] = x[i];
}
//-----------------------------------------------------------------------------
FacetArea::FacetArea(std::shared_ptr<const Mesh> mesh)
  : _mesh(mesh),
    not_on_boundary("*** Warning: evaluating special function FacetArea on a "
                    "non-facet domain, returning zero.")
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetArea::eval(Array<double>& values, const Array<double>& x,
                     const ufc::cell& cell) const
{
  dolfin_assert(_mesh);
  dolfin_assert(cell.geometric_dimension == _mesh->geometry().dim());

  if (cell.local_facet >= 0)
  {
    Cell c(*_mesh, cell.index);
    values[0] = c.facet_area(cell.local_facet);
  }
  else
  {
    not_on_boundary();
    values[0] = 0.0;
  }
}
//-----------------------------------------------------------------------------
