// Copyright (C) 2005-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-02-02
// Last changed: 2008-10-06

#include <dolfin/common/Array.h>
#include "ComplexODE.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
ComplexODE::ComplexODE(uint n, double T) : ODE(2*n, T), n(n), j(0.0, 1.0),
                                           zvalues(0), fvalues(0), yvalues(0)
{
  message("Creating complex ODE of size %d (%d complex components).", N, n);
  
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
  if ( zvalues ) delete [] zvalues;
  if ( fvalues ) delete [] fvalues;
  if ( yvalues ) delete [] yvalues;
}
//-----------------------------------------------------------------------------
complex ComplexODE::f(const complex z[], double t, uint i)
{
  error("Right-hand side for complex ODE not supplied by user.");
  
  complex zvalue;
  return zvalue;
}
//-----------------------------------------------------------------------------
void ComplexODE::f(const complex z[], double t, complex y[])
{
  // If a user of the mono-adaptive solver does not supply this function,
  // then call f() for each component.
 
  for (uint i = 0; i < n; i++)
    y[i] = this->f(z, t, i);
}
//-----------------------------------------------------------------------------
void ComplexODE::M(const complex x[], complex y[], const complex z[], double t)
{
  // Assume M is the identity if not supplied by user: y = x
  for (uint i = 0; i < n; i++)
    y[i] = x[i];
}
//-----------------------------------------------------------------------------
void ComplexODE::J(const complex x[], complex y[], const complex z[], double t)
{
  // If a user does not supply J, then compute it by the approximation
  //
  //     Jx = ( f(z + hx) - f(z - hx) ) / 2h

  error("Not implemented yet...");
}
//-----------------------------------------------------------------------------
double ComplexODE::k(uint i)
{
  return default_timestep;
}
//-----------------------------------------------------------------------------
bool ComplexODE::update(const complex z[], double t, bool end)
{
  return true;
}
//-----------------------------------------------------------------------------
void ComplexODE::u0(double* u)
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
double ComplexODE::f(const double* u, double t, uint i)
{
  // Translate right-hand side from complex to real, assuming that if
  // u_i depends on u_j, then u_i depends on both the double and
  // imaginary parts of the corresponding z_i

  // Update zvalues for correct components
  const Array<uint>& deps = dependencies[i];
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
void ComplexODE::f(const double* u, double t, double* y)
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
void ComplexODE::M(const double* x, double* y,
		   const double* u, double t)
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
void ComplexODE::J(const double* x, double* y,
		   const double* u, double t)
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
double ComplexODE::timestep(uint i)
{
  // Translate time step
  return k(i / 2);
}
//-----------------------------------------------------------------------------
bool ComplexODE::update(const double* u, double t, bool end)
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
