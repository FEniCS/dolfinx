// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewArray.h>
#include <dolfin/ComplexODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ComplexODE::ComplexODE(uint n) : ODE(2*n), n(n), j(0.0, 1.0),
				 zvalues(0), fvalues(0), yvalues(0)
{
  // Initialize complex solution vector and right-hand side
  zvalues = new complex[n];
  fvalues = new complex[n];
  for (uint i = 0; i < n; i++)
  {
    zvalues[i] = 0.0;
    fvalues[i] = 0.0;
  }

  dolfin_info("Creating complex ODE of size %d (%d complex components).", N, n);
}
//-----------------------------------------------------------------------------
ComplexODE::~ComplexODE()
{
  if ( zvalues ) delete [] zvalues;
  if ( fvalues ) delete [] fvalues;
  if ( yvalues ) delete [] yvalues;
}
//-----------------------------------------------------------------------------
complex ComplexODE::f(const complex z[], real t, uint i)
{
  dolfin_error("Not implemented.");
  
  complex zvalue;
  return zvalue;
}
//-----------------------------------------------------------------------------
void ComplexODE::feval(const complex z[], real t, complex f[])
{
  // If a user of the mono-adaptive solver does not supply this function,
  // then call f() for each component.
  
  for (uint i = 0; i < n; i++)
    f[i] = this->f(z, t, i);
}
//-----------------------------------------------------------------------------
void ComplexODE::M(const complex x[], complex y[], const complex z[], real t)
{
  // Assume M is the identity if not supplied by user: y = x

  for (uint i = 0; i < n; i++)
    y[i] = x[i];
}
//-----------------------------------------------------------------------------
real ComplexODE::k(uint i)
{
  return default_timestep;
}
//-----------------------------------------------------------------------------
void ComplexODE::update(const complex z[], real t)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real ComplexODE::u0(uint i)
{
  // Translate initial value from complex to real
  complex z = z0(i / 2);
  return ( i % 2 == 0 ? z.real() : z.imag() );
}
//-----------------------------------------------------------------------------
real ComplexODE::f(const real u[], real t, uint i)
{
  // Translate right-hand side from complex to real, assuming that if
  // u_i depends on u_j, then u_i depends on both the real and
  // imaginary parts of the corresponding z_i

  // Update zvalues for correct components
  const NewArray<uint>& deps = dependencies[i];
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
void ComplexODE::feval(const real u[], real t, real f[])
{
  // Update zvalues for all components
  for (uint i = 0; i < n; i++)
  {
    const complex zvalue(u[2*i], u[2*i + 1]);
    zvalues[i] = zvalue;
  }

  // Call user-supplied function feval(z, t, f)
  feval(zvalues, t, fvalues);
  
  // Translate values into f
  for (uint i = 0; i < n; i++)
  {
    const complex fvalue = fvalues[i];
    f[2*i] = fvalue.real();
    f[2*i + 1] = fvalue.imag();
  }
}
//-----------------------------------------------------------------------------
void ComplexODE::M(const real x[], real y[], const real u[], real t)
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
real ComplexODE::timestep(uint i)
{
  // Translate time step
  return k(i / 2);
}
//-----------------------------------------------------------------------------
void ComplexODE::update(const real u[], real t)
{
  // Update zvalues for all components
  for (uint i = 0; i < n; i++)
  {
    const complex zvalue(u[2*i], u[2*i + 1]);
    zvalues[i] = zvalue;
  }

  // Call user-supplied function update(z, t)
  update(zvalues, t);
}
//-----------------------------------------------------------------------------
