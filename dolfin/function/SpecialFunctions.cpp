// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007, 2008.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2008-07-17
// Last changed: 2009-10-05

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
void MeshCoordinates::eval(double* values, const Data& data) const
{
  error("MeshCoordinates::eval broken");
  /*
  assert(values);
  assert(data.geometric_dimension() == geometric_dimension());
  assert(data.x);

  for (uint i = 0; i < data.geometric_dimension(); ++i)
    values[i] = data.x[i];
  */
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
CellSize::CellSize(const Mesh& mesh)
  : mesh(mesh)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CellSize::eval(double* values, const Data& data) const
{
  assert(values);
  //assert(data.geometric_dimension() == geometric_dimension());
  assert(&data.cell().mesh() == &mesh);

  //const uint cell_index = data.ufc_cell().entity_indices[data.ufc_cell().topological_dimension][0];
  //Cell cell(mesh, cell_index);

  //values[0] = cell.diameter();
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
void FacetArea::eval(double* values, const Data& data) const
{
  assert(values);
  //assert(data.geometric_dimension() == geometric_dimension());
  assert(&data.cell().mesh() == &mesh);

  if (data.on_facet())
    values[0] = data.cell().facet_area(data.facet());
  else
    values[0] = 0.0;
}
//-----------------------------------------------------------------------------
