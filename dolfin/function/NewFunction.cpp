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
NewFunction::NewFunction(const FunctionSpace& V)
  : V(&V, NoDeleter<const FunctionSpace>()), x(0)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const std::tr1::shared_ptr<FunctionSpace> V)
  : V(V), x(0)
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
const FunctionSpace& NewFunction::function_space() const
{
  dolfin_assert(V);
  return *V;
}
//-----------------------------------------------------------------------------
GenericVector& NewFunction::vector()
{
  // Initialize vector of dofs if not initialized
  if (!x)
    init();

  dolfin_assert(x);
  return *x;
}
//-----------------------------------------------------------------------------
const GenericVector& NewFunction::vector() const
{
  // Check if vector of dofs has been initialized
  if (!x)
    error("Requesting vector of degrees of freedom for function, but vector has not been initialized.");

  dolfin_assert(x);
  return *x;
}
//-----------------------------------------------------------------------------
void NewFunction::eval(real* values, const real* p) const
{
  dolfin_assert(values);
  dolfin_assert(p);
  dolfin_assert(V);

  // Use vector of dofs if available
  if (this->x)
  {
    V->eval(values, p, *x);
    return;
  }

  // Use scalar eval() if available
  if (V->element().value_rank() == 0)
  {
    values[0] = eval(p);
    return;
  }

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
  dolfin_assert(V);
  const uint N = V->dofmap().global_dimension();

  // Create vector of dofs
  if (!x)
  {
    DefaultFactory factory;
    x = factory.create_vector();
  }

  // Initialize vector of dofs
  dolfin_assert(x);
  x->resize(N);
  x->zero();
}
//-----------------------------------------------------------------------------
