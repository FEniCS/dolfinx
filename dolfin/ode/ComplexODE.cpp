// Copyright (C) 2005-2008 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-02-02
// Last changed: 2011-03-17

#ifndef HAS_GMP

#include "ComplexODE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ComplexODE::ComplexODE(uint n, real T) : ODE(2*n, T), n(n), j(0.0, 1.0),
                                           zvalues(0), fvalues(0), yvalues(0)
{
  log(TRACE, "Creating complex ODE of size %d (%d complex components).", N, n);

  // Initialize complex solution vector and right-hand side
  zvalues = new complex[n];
  fvalues = new complex[n];
  for (uint i = 0; i < n; i++)
  {
    zvalues[i] = 0.0;
    fvalues[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
ComplexODE::~ComplexODE()
{
  delete [] zvalues;
  delete [] fvalues;
  delete [] yvalues;
}
//-----------------------------------------------------------------------------
complex ComplexODE::f(const complex z[], real t, uint i)
{
  error("Right-hand side for complex ODE not supplied by user.");

  complex zvalue;
  return zvalue;
}
//-----------------------------------------------------------------------------
void ComplexODE::f(const complex z[], real t, complex y[])
{
  // If a user of the mono-adaptive solver does not supply this function,
  // then call f() for each component.

  for (uint i = 0; i < n; i++)
    y[i] = this->f(z, t, i);
}
//-----------------------------------------------------------------------------
void ComplexODE::M(const complex x[], complex y[], const complex z[], real t)
{
  // Assume M is the identity if not supplied by user: y = x
  for (uint i = 0; i < n; i++)
    y[i] = x[i];
}
//-----------------------------------------------------------------------------
void ComplexODE::J(const complex x[], complex y[], const complex z[], real t)
{
  // If a user does not supply J, then compute it by the approximation
  //
  //     Jx = ( f(z + hx) - f(z - hx) ) / 2h

  dolfin_not_implemented();
}
//-----------------------------------------------------------------------------
real ComplexODE::k(uint i)
{
  return default_timestep;
}
//-----------------------------------------------------------------------------
bool ComplexODE::update(const complex z[], real t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------
void ComplexODE::u0(Array<real>& u)
{
  // Translate initial value from complex to real
  z0(zvalues);
  for (uint i = 0; i < N; i++)
  {
    complex z = zvalues[i / 2];
    u[i] = ( i % 2 == 0 ? z.real() : z.imag() );
  }
}
//-----------------------------------------------------------------------------
real ComplexODE::f(const Array<real>& u, real t, uint i)
{
  // Translate right-hand side from complex to real, assuming that if
  // u_i depends on u_j, then u_i depends on both the real and
  // imaginary parts of the corresponding z_i

  // Update zvalues for correct components
  const std::vector<uint>& deps = dependencies[i];
  for (uint pos = 0; pos < deps.size(); pos++)
  {
    // Element of u that needs to be updated
    const uint ju = deps[pos];

    // Corresponding element of z
    const uint jz = ju / 2;

    // Update value
    const complex zvalue(u[2*jz], u[2*jz + 1]);
    zvalues[jz] = zvalue;
  }

  // Call user-supplied function f(z, t, i)
  const complex fvalue = f(zvalues, t, i / 2);

  // Return value
  return ( i % 2 == 0 ? fvalue.real() : fvalue.imag() );
}
//-----------------------------------------------------------------------------
void ComplexODE::f(const Array<real>& u, real t, Array<real>& y)
{
  // Update zvalues for all components
  for (uint i = 0; i < n; i++)
  {
    const complex zvalue(u[2*i], u[2*i + 1]);
    zvalues[i] = zvalue;
  }

  // Call user-supplied function f(z, t, y)
  f(zvalues, t, fvalues);

  // Translate values into f
  for (uint i = 0; i < n; i++)
  {
    const complex fvalue = fvalues[i];
    y[2*i] = fvalue.real();
    y[2*i + 1] = fvalue.imag();
  }
}
//-----------------------------------------------------------------------------
void ComplexODE::M(const Array<real>& x, Array<real>& y,
		   const Array<real>& u, real t)
{
  // Update zvalues and fvalues for all components
  for (uint i = 0; i < n; i++)
  {
    const complex zvalue(u[2*i], u[2*i + 1]);
    const complex xvalue(x[2*i], x[2*i + 1]);
    zvalues[i] = zvalue;
    fvalues[i] = xvalue; // Use fvalues for x
  }

  // Use additional array for y, initialize the first time
  if ( !yvalues )
  {
    yvalues = new complex[n];
    for (uint i = 0; i < n; i++)
      yvalues[i] = 0.0;
  }

  // Call user-supplied function M(x, y, z, t)
  M(fvalues, yvalues, zvalues, t);

  // Copy values to y
  for (uint i = 0; i < n; i++)
  {
    const complex yvalue = yvalues[i];
    y[2*i] = yvalue.real();
    y[2*i + 1] = yvalue.imag();
  }
}
//-----------------------------------------------------------------------------
void ComplexODE::J(const Array<real>& x, Array<real>& y,
		   const Array<real>& u, real t)
{
  // Update zvalues and fvalues for all components
  for (uint i = 0; i < n; i++)
  {
    const complex zvalue(u[2*i], u[2*i + 1]);
    const complex xvalue(x[2*i], x[2*i + 1]);
    zvalues[i] = zvalue;
    fvalues[i] = xvalue; // Use fvalues for x
  }

  // Use additional array for y, initialize the first time
  if ( !yvalues )
  {
    yvalues = new complex[n];
    for (uint i = 0; i < n; i++)
      yvalues[i] = 0.0;
  }

  // Call user-supplied function J(x, y, z, t)
  J(fvalues, yvalues, zvalues, t);

  // Copy values to y
  for (uint i = 0; i < n; i++)
  {
    const complex yvalue = yvalues[i];
    y[2*i] = yvalue.real();
    y[2*i + 1] = yvalue.imag();
  }
}
//-----------------------------------------------------------------------------
real ComplexODE::timestep(uint i)
{
  // Translate time step
  return k(i / 2);
}
//-----------------------------------------------------------------------------
bool ComplexODE::update(const Array<real>& u, real t, bool end)
{
  // Update zvalues for all components
  for (uint i = 0; i < n; i++)
  {
    const complex zvalue(u[2*i], u[2*i + 1]);
    zvalues[i] = zvalue;
  }

  // Call user-supplied function update(z, t)
  return update(zvalues, t, end);
}
//-----------------------------------------------------------------------------
#endif
