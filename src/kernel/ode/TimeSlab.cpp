// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-05-02
// Last changed: 2006-08-21

#include <stdio.h>
#include <string>
#include <dolfin/ParameterSystem.h>
#include <dolfin/ODE.h>
#include <dolfin/cGqMethod.h>
#include <dolfin/dGqMethod.h>
#include <dolfin/TimeSlab.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlab::TimeSlab(ODE& ode) : 
  N(ode.size()), _a(0.0), _b(0.0), ode(ode), method(0), u0(ode.size()),
  save_final(get("ODE save final solution"))
{
  // Choose method
  std::string m = get("ODE method");
  int q = get("ODE order");
  if ( m == "cg" || m == "mcg" )
  {
    if ( q < 1 )
      dolfin_error("Minimal order is q = 1 for continuous Galerkin.");
    method = new cGqMethod(q);
  }
  else if ( m == "dg" || m == "mdg" )
  {
    if ( q < 0 )
      dolfin_error("Minimal order is q = 0 for discontinuous Galerkin.");
    method = new dGqMethod(q);
  }
  else
    dolfin_error1("Unknown ODE method: %s", m.c_str());

  // Get initial data
  u0 = 0.0;
  ode.u0(u0);
}
//-----------------------------------------------------------------------------
TimeSlab::~TimeSlab()
{
  if ( method ) delete method;
}
//-----------------------------------------------------------------------------
dolfin::uint TimeSlab::size() const
{
  return N;
}
//-----------------------------------------------------------------------------
real TimeSlab::starttime() const
{
  return _a;
}
//-----------------------------------------------------------------------------
real TimeSlab::endtime() const
{
  return _b;
}
//-----------------------------------------------------------------------------
real TimeSlab::length() const
{
  return _b - _a;
}
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<<(LogStream& stream,
				      const TimeSlab& timeslab)
{
  stream << "[ TimeSlab of length " << timeslab.length()
	 << " between a = " << timeslab.starttime()
	 << " and b = " << timeslab.endtime() << " ]";
  
  return stream;
}
//-----------------------------------------------------------------------------
void TimeSlab::write(const uBlasVector& u)
{
  // FIXME: Make this a parameter?
  std::string filename = "solution.data";
  dolfin_info("Saving solution at final time to file \"%s\".",
	      filename.c_str());

  FILE* fp = fopen(filename.c_str(), "w");
  for (uint i = 0; i < u.size(); i++)
  {
    fprintf(fp, "%.15e ", u(i));
  }
  fprintf(fp, "\n");
  fclose(fp);
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const real x[], uint xoffset, real y[], uint yoffset, uint n)
{
  for (uint i = 0; i < n; i++)
    y[yoffset + i] = x[xoffset + i];
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const uBlasVector& x, uint xoffset, real y[], uint yoffset, uint n)
{
  for (uint i = 0; i < n; i++)
    y[yoffset + i] = x(xoffset + i);
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const real x[], uint xoffset, uBlasVector& y, uint yoffset, uint n)
{
  for (uint i = 0; i < n; i++)
    y(yoffset + i) = x[xoffset + i];
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const uBlasVector& x, uint xoffset, uBlasVector& y, uint yoffset, uint n)
{
  for (uint i = 0; i < n; i++)
    y(yoffset + i) = x(xoffset + i);
}
//-----------------------------------------------------------------------------
