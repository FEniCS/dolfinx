// Copyright (C) 2006-2009 Anders Logg
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
// Last changed: 2009-11-14

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include "SpecialFunctions.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshCoordinates::MeshCoordinates(const Mesh& mesh)
  : Expression(mesh.geometry().dim()), mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshCoordinates::eval(Array<double>& values, const Array<double>& x,
                           const ufc::cell& cell) const
{
  assert(cell.geometric_dimension == mesh.geometry().dim());
  assert(x.size() == mesh.geometry().dim());

  for (uint i = 0; i < cell.geometric_dimension; ++i)
    values[i] = x[i];
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
/*
CellSize::CellSize(const Mesh& mesh)
  : mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CellSize::eval(Array<double>& values, const Data& data) const
{
  assert(&data.cell().mesh() == &mesh);
  values[0] = data.cell().diameter();
}
*/
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
FacetArea::FacetArea(const Mesh& mesh)
  : mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetArea::eval(Array<double>& values, const Array<double>& x,
                     const ufc::cell& cell) const
{
  error("FacetArea::eval in SpecialFunctions needs to be updated");
  /*
  assert(&data.cell().mesh() == &mesh);

  if (data.on_facet())
    values[0] = data.cell().facet_area(data.facet());
  else
    values[0] = 0.0;
  */
}
//-----------------------------------------------------------------------------
