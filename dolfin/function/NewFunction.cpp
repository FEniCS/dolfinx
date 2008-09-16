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
  dolfin_assert(_U);
  return *_U;
}
//-----------------------------------------------------------------------------
const GenericVector& NewFunction::U() const
{
  dolfin_assert(_U);
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
