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
const NewFunction& NewFunction::operator= (const NewFunction& v)
{
  // FIXME: Need to check pointers here and check if _U is nonzero
  //*_V = *v._V;
  //*_U = *u._V;
  
  return *this;
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
