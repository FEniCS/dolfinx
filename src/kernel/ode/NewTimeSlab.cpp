// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <string>
#include <dolfin/dolfin_settings.h>
#include <dolfin/ODE.h>
#include <dolfin/NewcGqMethod.h>
#include <dolfin/NewdGqMethod.h>
#include <dolfin/NewTimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewTimeSlab::NewTimeSlab(ODE& ode) : 
  N(ode.size()), _a(0.0), _b(0.0), ode(ode), method(0), u0(0)
{
  // Choose method
  std::string m = dolfin_get("method");
  int q = dolfin_get("order");
  if ( m == "cg" )
    method = new NewcGqMethod(q);
  else
    method = new NewdGqMethod(q);

  // Initialize initial data
  u0 = new real[N];

  // Get initial data
  for (uint i = 0; i < N; i++)
  {
    u0[i] = ode.u0(i);
    //cout << "u0[" << i << "] = " << u0[i] << endl;
  }
}
//-----------------------------------------------------------------------------
NewTimeSlab::~NewTimeSlab()
{
  if ( method ) delete method;
  if ( u0 ) delete [] u0;
}
//-----------------------------------------------------------------------------
dolfin::uint NewTimeSlab::size() const
{
  return N;
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
dolfin::LogStream& dolfin::operator<<(LogStream& stream,
				      const NewTimeSlab& timeslab)
{
  stream << "[ TimeSlab of length " << timeslab.length()
	 << " between a = " << timeslab.starttime()
	 << " and b = " << timeslab.endtime() << " ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
