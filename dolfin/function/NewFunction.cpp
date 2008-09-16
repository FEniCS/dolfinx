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
  : _V(&V, NoDeleter<FunctionSpace>()),
    _U(std::tr1::shared_ptr<GenericVector>())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(std::tr1::shared_ptr<FunctionSpace> V)
  : _V(V),
    _U(std::tr1::shared_ptr<GenericVector>())
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const std::string filename)
{
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
NewFunction::NewFunction(const NewFunction& f)
{
  error("Not implemented.");
}
//-----------------------------------------------------------------------------
NewFunction::~NewFunction()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
FunctionSpace& NewFunction::V()
{
  dolfin_assert(_V);
  return *_V;
}
//-----------------------------------------------------------------------------
const FunctionSpace& NewFunction::V() const
{
  dolfin_assert(_V);
  return *_V;
}
//-----------------------------------------------------------------------------
GenericVector& NewFunction::U()
{
  // Initialize vector if not initialized
  if (!_U)
    init();

  return *_U;
}
//-----------------------------------------------------------------------------
const GenericVector& NewFunction::U() const
{
  // Check if vector has been initialized
  if (!_U)
    error("Requesting vector of degrees of freedom for function, but vector has not been initialized.");

  return *_U;
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
  const uint N = _V->dofmap().global_dimension();

  // Create vector
  if (!_U)
  {
    DefaultFactory factory;
    _U = std::tr1::shared_ptr<GenericVector>(factory.createVector());
  }

  // Initialize vector
  _U->init(N);
}
//-----------------------------------------------------------------------------
