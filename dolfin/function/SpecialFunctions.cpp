// Copyright (C) 2006-2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Kristian B. Oelgaard, 2007, 2008.
// Modified by Martin Sandve Alnes, 2008.
// Modified by Garth N. Wells, 2008.
//
// First added:  2008-07-17
// Last changed: 2009-03-11

#include <cmath>

#include <dolfin/common/constants.h>
#include <dolfin/mesh/Mesh.h>
#include <dolfin/mesh/Facet.h>
#include <dolfin/mesh/Point.h>
#include <dolfin/mesh/Vertex.h>
#include <dolfin/fem/Form.h>
#include <dolfin/fem/FiniteElement.h>
#include "FunctionSpace.h"
#include "SpecialFunctions.h"
#include "Data.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
MeshCoordinates::MeshCoordinates() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MeshCoordinates::MeshCoordinates(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MeshCoordinates::eval(double* values, const Data& data) const
{
  for (uint i = 0; i < data.geometric_dimension(); i++)
    values[i] = data.x[i];
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
CellSize::CellSize() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellSize::CellSize(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void CellSize::eval(double* values, const Data& data) const
{
  values[0] = data.cell().diameter();
}
//-----------------------------------------------------------------------------
double CellSize::min() const
{
  CellIterator c(function_space().mesh());
  double hmin = c->diameter();
  for (; !c.end(); ++c)
    hmin = std::min(hmin, c->diameter());
  return hmin;
}
//-----------------------------------------------------------------------------
double CellSize::max() const
{
  CellIterator c(function_space().mesh());
  double hmax = c->diameter();
  for (; !c.end(); ++c)
    hmax = std::max(hmax, c->diameter());
  return hmax;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
FacetNormal::FacetNormal() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FacetNormal::FacetNormal(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetNormal::eval(double* values, const Data& data) const
{
  const Cell& cell = data.cell();

  if (data.on_facet())
  {
    const uint facet = data.facet();
    for (uint i = 0; i < cell.dim(); i++)
      values[i] = cell.normal(facet, i);
  }
  else
  {
    for (uint i = 0; i < cell.dim(); i++)
      values[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
dolfin::uint FacetNormal::rank() const
{
  return 1;
}
//-----------------------------------------------------------------------------
dolfin::uint FacetNormal::dim(uint i) const
{
  if(i > 0)
    error("Invalid dimension %d in FacetNormal::dim.", i);
  return function_space().mesh().geometry().dim();
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
FacetArea::FacetArea() : Function()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FacetArea::FacetArea(const FunctionSpace& V) : Function(V)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void FacetArea::eval(double* values, const Data& data) const
{
  if (data.on_facet())
    values[0] = data.cell().facet_area(data.facet());
  else
    values[0] = 0.0;
}
//-----------------------------------------------------------------------------
//-----------------------------------------------------------------------------
SUPGStabilizer::SUPGStabilizer(const FunctionSpace& V, const Function& f, double sigma_)
  : Function(V), sigma(sigma_), field(&f)
{
  // Some simple sanity checks on function
  if (&V.mesh() != &f.function_space().mesh())
    error("The mesh provided with the FunctionSpace must be the same as the one in the provided field Function.");

  if (f.function_space().element().value_rank() != 1)
    error("The provided field function need to be a vector valued function.");

  const uint geometric_dimension =  _function_space->mesh().geometry().dim();
  if (f.function_space().element().value_dimension(0) != geometric_dimension)
    error("The provided field function value dimension need to be the same as the geometric dimension.");

  if (sigma_ < 0.0)
    error("Provide a positive value for sigma");
}
//-----------------------------------------------------------------------------
void SUPGStabilizer::eval(double* values, const Data& data) const
{
  assert(values);
  assert(field);
  const uint geometric_dimension =  _function_space->mesh().geometry().dim();
  double field_norm = 0.0;
  double tau = 0.0;
  const double h = data.cell().diameter();
  UFCCell ufc_cell(data.cell());


  // Evaluate the advective field
  field->eval(values, data.x, ufc_cell, data.cell().index());

  // Compute the norm of the field
  for (uint i = 0; i < geometric_dimension; ++i)
    field_norm += values[i]*values[i];
  field_norm = sqrt(field_norm);

  // Local Péclet number
  const double PE = 0.5*field_norm*h/sigma;

  // Compute the local stabilizing factor tau
  if (PE > DOLFIN_EPS)
    tau = 1.0/std::tanh(PE)-1.0/PE;

  // Weight the field with the norm, together with the cell size and
  // the local stabilizing factor
  for (uint i = 0; i < geometric_dimension; ++i)
    values[i] *= 0.5*h*tau/field_norm;
}
//-----------------------------------------------------------------------------
