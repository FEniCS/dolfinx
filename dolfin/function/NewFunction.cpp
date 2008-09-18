// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2005-2008.
// Modified by Martin Sandve Alnes, 2008.
//
// First added:  2003-11-28
// Last changed: 2008-09-11

#include <dolfin/log/log.h>
#include <dolfin/common/NoDeleter.h>
#include <dolfin/la/GenericVector.h>
#include <dolfin/la/DefaultFactory.h>
#include <dolfin/fem/FiniteElement.h>
#include <dolfin/fem/DofMap.h>
#include "FunctionSpace.h"
#include "NewFunction.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
NewFunction::NewFunction(FunctionSpace& V)
  : V(&V, NoDeleter<FunctionSpace>()),
    x(std::tr1::shared_ptr<GenericVector>())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(std::tr1::shared_ptr<FunctionSpace> V)
  : V(V),
    x(std::tr1::shared_ptr<GenericVector>())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const std::string filename)
{
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const NewFunction& v)
{
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
NewFunction::~NewFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const NewFunction& NewFunction::operator= (const NewFunction& v)
{
  // FIXME: Need to check pointers here and check if _U is nonzero
  //*_V = *v._V;
  //*_U = *u._V;
  
  return *this;
}
//-----------------------------------------------------------------------------
FunctionSpace& NewFunction::function_space()
{
  dolfin_assert(V);
  return *V;
}
//-----------------------------------------------------------------------------
const FunctionSpace& NewFunction::function_space() const
{
  dolfin_assert(V);
  return *V;
}
//-----------------------------------------------------------------------------
GenericVector& NewFunction::vector()
{
  // Initialize vector if not initialized
  if (!x)
    init();

  return *x;
}
//-----------------------------------------------------------------------------
const GenericVector& NewFunction::vector() const
{
  // Check if vector has been initialized
  if (!x)
    error("Requesting vector of degrees of freedom for function, but vector has not been initialized.");

  return *x;
}
//-----------------------------------------------------------------------------
void NewFunction::eval(real* values, const real* x) const
{
  // Check if we have a vector of degrees of freedom
  if (x)
  {
    /*
    // Find the cell that contains x
    const uint gdim = mesh->geometry().dim();
    if (gdim > 3)
      error("Sorry, point evaluation of functions not implemented for meshes of dimension %d.", gdim);
    Point p;
    for (uint i = 0; i < gdim; i++)
      p[i] = x[i];
    Array<uint> cells;
    intersection_detector->overlap(p, cells);
    if (cells.size() < 1)
      error("Unable to evaluate function at given point (not inside domain).");
    Cell cell(*mesh, cells[0]);
    UFCCell ufc_cell(cell);
  
    // Get expansion coefficients on cell
    this->interpolate(scratch->coefficients, ufc_cell, *finite_element);

    // Compute linear combination
    for (uint j = 0; j < scratch->size; j++)
      values[j] = 0.0;
    for (uint i = 0; i < finite_element->spaceDimension(); i++)
    {
      finite_element->evaluateBasis(i, scratch->values, x, ufc_cell);
      for (uint j = 0; j < scratch->size; j++)
        values[j] += scratch->coefficients[i] * scratch->values[j];
    }
    */
  }

  // Try calling scalar eval() for scalar-valued function
  if (V->element().valueRank() == 0)
    values[0] = eval(x);

  // Missing eval function
  error("Missing eval() for user-defined function (must be overloaded).");
}
//-----------------------------------------------------------------------------
dolfin::real NewFunction::eval(const real* x) const
{
  // Missing eval function
  error("Missing eval() for user-defined function (must be overloaded).");
  return 0.0;
}
//-----------------------------------------------------------------------------
void NewFunction::eval(simple_array<real>& values, const simple_array<real>& x) const
{
  eval(values.data, x.data);
}
//-----------------------------------------------------------------------------
void NewFunction::init()
{
  // Get size
  const uint N = V->dofmap().global_dimension();

  // Create vector
  if (!x)
  {
    DefaultFactory factory;
    x = std::tr1::shared_ptr<GenericVector>(factory.createVector());
  }

  // Initialize vector
  x->init(N);
}
//-----------------------------------------------------------------------------
