// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewArray.h>
#include <dolfin/ComplexODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ComplexODE::ComplexODE(uint n) : ODE(2*n), zvalues(0), fvalues(0)
{
  // Initialize complex solution vector and right-hand side
  zvalues = new complex[n];
  fvalues = new complex[n];
  for (uint i = 0; i < n; i++)
  {
    zvalues[i] = 0.0;
    fvalues[i] = 0.0;
  }

  dolfin_info("Creating complex ODE of size %d (%d complex compoents).", N, n);
}
//-----------------------------------------------------------------------------
ComplexODE::~ComplexODE()
{
  if ( zvalues ) delete [] zvalues;
  if ( fvalues ) delete [] fvalues;
}
//-----------------------------------------------------------------------------
complex ComplexODE::f(complex z[], real t, uint i)
{
  dolfin_error("Not implemented.");
  
  complex zvalue;
  return zvalue;
}
//-----------------------------------------------------------------------------
void ComplexODE::feval(complex z[], real t, complex f[])
{
  // If a user of the mono-adaptive solver does not supply this function,
  // then call f() for each component.
  
  for (uint i = 0; i < n; i++)
    f[i] = this->f(z, t, i);
}
//-----------------------------------------------------------------------------
real ComplexODE::k(uint i)
{
  return default_timestep;
}
//-----------------------------------------------------------------------------
real ComplexODE::u0(uint i)
{
  // Translate initial value from complex to real
  complex z = z0(i / 2);
  return ( i % 2 == 0 ? z.real() : z.imag() );
}
//-----------------------------------------------------------------------------
real ComplexODE::f(real u[], real t, uint i)
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
void ComplexODE::feval(real u[], real t, real f[])
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
real ComplexODE::timestep(uint i)
{
  // Translate time step
  return k(i / 2);
}
//-----------------------------------------------------------------------------
