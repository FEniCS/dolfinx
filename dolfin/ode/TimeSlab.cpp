// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-05-02
// Last changed: 2009-09-08

#include <stdio.h>
#include <string>
#include <dolfin/la/uBLASVector.h>
#include "ODE.h"
#include "cGqMethod.h"
#include "dGqMethod.h"
#include "TimeSlab.h"
#include <iostream>
#include <fstream>
#include <iomanip>


using namespace dolfin;

//-----------------------------------------------------------------------------
TimeSlab::TimeSlab(ODE& ode) :
  N(ode.size()), _a(0.0), _b(0.0), ode(ode), method(0), u0(N),
  save_final(ode.parameters["save_final_solution"])
{
  // Choose method
  std::string m = ode.parameters["method"];
  int q = ode.parameters["order"];
  if ( m == "cg" || m == "mcg" )
  {
    if ( q < 1 )
      error("Minimal order is q = 1 for continuous Galerkin.");
    method = new cGqMethod(q);
  }
  else if ( m == "dg" || m == "mdg" )
  {
    if ( q < 0 )
      error("Minimal order is q = 0 for discontinuous Galerkin.");
    method = new dGqMethod(q);
  }
  else
    error("Unknown ODE method: %s", m.c_str());

  // Get iinitial data
  //u0 = new real[ode.size()];
  //real_zero(ode.size(), u0);
  ode.u0(u0);
}
//-----------------------------------------------------------------------------
TimeSlab::~TimeSlab()
{
  delete method;
  //delete [] u0;
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
void TimeSlab::set_state(const real* u)
{
  for (uint i = 0; i < N; i++)
    u0[i] = u[i];
}
//-----------------------------------------------------------------------------
void TimeSlab::get_state(real* u)
{
  for (uint i = 0; i < N; i++)
    u[i] = u0[i];
}
//-----------------------------------------------------------------------------
const Lagrange TimeSlab::get_trial() const {
  return method->get_trial();
}
//-----------------------------------------------------------------------------
const real* TimeSlab::get_quadrature_weights() const {
  return method->get_quadrature_weights();
}
//-----------------------------------------------------------------------------
void TimeSlab::write(Array<real>& u)
{
  // FIXME: Make this a parameter?
  std::string filename = "solution.data";

  info("Saving solution at final time to file \"%s\".", filename.c_str());

  std::ofstream fp(filename.c_str());
  if (!fp.is_open())
    error("Unable to open file %s", filename.c_str());

  fp << std::setprecision(real_decimal_prec());

  for (uint i = 0; i < u.size(); i++)
  {
    fp << u[i] << " ";
  }
  fp << std::endl;
  fp.close();
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const real x[], uint xoffset, real y[], uint yoffset, uint n)
{
  for (uint i = 0; i < n; i++)
    y[yoffset + i] = x[xoffset + i];
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const uBLASVector& x, uint xoffset, real y[], uint yoffset, uint n)
{
  for (uint i = 0; i < n; i++)
  {
    y[yoffset + i] = x[xoffset + i];
  }
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const real x[], uint xoffset, uBLASVector& y, uint yoffset, uint n)
{
  for (uint i = 0; i < n; i++)
    //Note: Precision lost if working with GMP
    y[yoffset + i] = to_double(x[xoffset + i]);

}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const uBLASVector& x, uint xoffset, uBLASVector& y, uint yoffset, uint n)
{
  for (uint i = 0; i < n; i++)
    y[yoffset + i] = x[xoffset + i];
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const uBLASVector& x, uint xoffset, Array<real>& y)
{
  for (uint i = 0; i < y.size(); i++)
    y[i] = x[xoffset + i];
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const real* x, uint xoffset, Array<real>& y)
{
  for (uint i = 0; i < y.size(); i++)
    y[i] = x[xoffset + i];
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const Array<real>& x, uBLASVector& y , uint yoffset)
{
  for (uint i = 0; i < x.size(); i++)
    y[yoffset + i] = to_double(x[i]);
}
//-----------------------------------------------------------------------------
void TimeSlab::copy(const Array<real>& x, real* y, uint yoffset)
{
  for (uint i = 0; i < x.size(); i++)
    y[yoffset + i] = to_double(x[i]);
}

/*
void TimeSlab::copy(const uBLASVector& x, uint xoffset, Array<real>& y)
{
  for (uint i = 0; i < y.size(); i++)
    y[i] = x[xoffset + i];
}
*/
