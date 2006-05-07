// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005
// Last changed: 2006-05-07

#ifdef HAVE_PETSC_H

#include <dolfin/dolfin_log.h>
#include <dolfin/Homotopy.h>
#include <dolfin/HomotopyODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
HomotopyODE::HomotopyODE(Homotopy& homotopy, uint n, real T)
  : ComplexODE(n, T), homotopy(homotopy), n(n), _state(ode), tmp(0)
{
  tmp = new complex[n];
  for (uint i = 0; i < n; i++)
    tmp[i] = 0.0;
}
//-----------------------------------------------------------------------------
HomotopyODE::~HomotopyODE()
{
  if ( tmp ) delete [] tmp;
}
//-----------------------------------------------------------------------------
complex HomotopyODE::z0(unsigned int i)
{
  const complex z = homotopy.z0(i);

  //cout << "Starting point: z = " << z << endl;

  return z;
}
//-----------------------------------------------------------------------------  
void HomotopyODE::f(const complex z[], real t, complex y[])
{
  // Need to compute f(z) = G(z) - F(z)

  // Call user-supplied function to compute F(z)
  homotopy.F(z, y);

  // Just compute F(z) if playing end game
  if ( _state == endgame )
    return;

  // Call possibly user-supplied function to compute G(z)
  homotopy.G(z, tmp);

  // Negate F and add G
  for (uint i = 0; i < n; i++)
    y[i] = tmp[i] - y[i];
}
//-----------------------------------------------------------------------------
void HomotopyODE::M(const complex x[], complex y[], const complex z[], real t)
{
  // Need to compute ((1 - t) G' + t F')*x

  // Call user-supplied function to compute F'*x
  homotopy.JF(z, x, y);

  // Call possible user-supplied function to compute G'*x
  homotopy.JG(z, x, tmp);

  // Add terms
  for (uint i = 0; i < n; i++)
    y[i] = (1.0 - t)*tmp[i] + t*y[i];
}
//-----------------------------------------------------------------------------
void HomotopyODE::J(const complex x[], complex y[], const complex z[], real t)
{
  // Need to compute (G' - F')*x

  // Call user-supplied function to compute F'*x
  homotopy.JF(z, x, y);
  
  // Just compute dF/dz if playing end game
  if ( _state == endgame )
    return;

  // Call possible user-supplied function to compute G'*x
  homotopy.JG(z, x, tmp);

  // Negate F'*x and add G'*x
  for (uint i = 0; i < n; i++)
    y[i] = tmp[i] - y[i];
}
//-----------------------------------------------------------------------------
bool HomotopyODE::update(const complex z[], real t, bool end)
{
  // FIXME: Maybe this should be a parameter?
  const real epsilon = 0.5;

  // Check if we should monitor the homotopy
  if ( homotopy.monitor )
    monitor(z, t);
  
  // Check convergence of (1 - t)*G(z)
  homotopy.G(z, tmp);
  for (uint i = 0; i < n; i++)
  {
    real r = std::abs(pow((1.0 - t), 1.0 - epsilon) * tmp[i]);
    //cout << "checking: r = " << r << endl;
    
    if ( r > homotopy.divtol )
    {
      dolfin_info("Homotopy path seems to be diverging.");
      return false;
    }
  }

  // Check if we reached the end of the integration
  if ( end )
  {
    dolfin_info("Reached end of integration, saving solution.");
    _state = endgame;
    
    real* xx = homotopy.x.array();
    for (uint i = 0; i < n; i++)
    {
      const complex zi = z[i];
      xx[2*i] = zi.real();
      xx[2*i + 1] = zi.imag();
    }
    homotopy.x.restore(xx);
  }
  
  return true;
}
//-----------------------------------------------------------------------------
HomotopyODE::State HomotopyODE::state()
{
  return _state;
}
//-----------------------------------------------------------------------------
void HomotopyODE::monitor(const complex z[], real t)
{
  dolfin_assert(tmp);

  // Monitor homotopy H(z, t) = t*F(z) + (1 - t)*G(z)

  // Temporary storage for h
  complex* h = new complex[n];

  // Evaluate F
  homotopy.F(z, tmp);
  real Fnorm = 0.0;
  for (uint i = 0; i < n; i++)
  {
    h[i] = t*tmp[i];
    Fnorm = std::max(Fnorm, std::abs(tmp[i]));
  }

  // Evaluate G
  homotopy.G(z, tmp);
  real Gnorm = 0.0;
  for (uint i = 0; i < n; i++)
  {
    h[i] += (1.0 - t)*tmp[i];
    Gnorm = std::max(Gnorm, std::abs(tmp[i]));
  }
  
  // Compute norm of H
  real Hnorm = 0.0;
  for (uint i = 0; i < n; i++)
    Hnorm = std::max(Hnorm, std::abs(h[i]));
  
  // Clear temporary storagexs
  delete [] h;

  dolfin_info("Homotopy: F = %e G = %e H = %e", Fnorm, Gnorm, Hnorm);
}
//-----------------------------------------------------------------------------

#endif
