// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007, 2008.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008, 2009.
//
// First added:  2008-07-17
// Last changed: 2009-11-14

#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Cell.h>
#include "Data.h"
#include "SpecialFunctions.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshCoordinates::MeshCoordinates(const Mesh& mesh)
  : Expression(mesh.geometry().dim()), mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshCoordinates::eval(Array<double>& values, const Data& data) const
{
  assert(data.geometric_dimension() == mesh.geometry().dim());
  assert(data.x.size() == mesh.geometry().dim());

  for (uint i = 0; i < data.geometric_dimension(); ++i)
    values[i] = (data.x)[i];
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
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
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
FacetArea::FacetArea(const Mesh& mesh)
  : mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetArea::eval(Array<double>& values, const Data& data) const
{
  assert(&data.cell().mesh() == &mesh);

  if (data.on_facet())
    values[0] = data.cell().facet_area(data.facet());
  else
    values[0] = 0.0;
}
//-----------------------------------------------------------------------------
