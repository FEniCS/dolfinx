// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/NewcGqMethod.h>
#include <dolfin/NewdGqMethod.h>
#include <dolfin/NewTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewTimeSlab::NewTimeSlab(ODE& ode) : 
  N(ode.size()), _a(0.0), _b(0.0), ode(ode), method(0)
{
  // Choose method
  std::string m = dolfin_get("method");
  int q = dolfin_get("order");
  if ( m == "cg" )
    method = new NewcGqMethod(q);
  else
    method = new NewdGqMethod(q);
}
//-----------------------------------------------------------------------------
NewTimeSlab::~NewTimeSlab()
{
  if ( method ) delete method;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::starttime() const
{
  return _a;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::endtime() const
{
  return _b;
}
//-----------------------------------------------------------------------------
real NewTimeSlab::length() const
{
  return _b - _a;
}
//-----------------------------------------------------------------------------
