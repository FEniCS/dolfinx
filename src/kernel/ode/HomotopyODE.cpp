// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_settings.h>
#include <dolfin/Homotopy.h>
#include <dolfin/HomotopyODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
HomotopyODE::HomotopyODE(Homotopy& homotopy, uint n)
  : ComplexODE(n), homotopy(homotopy), n(n), _state(ode), tmp(0)
{
  if ( homotopy.monitor )
  {
    tmp = new complex[n];
    for (uint i = 0; i < n; i++)
      tmp[i] = 0.0;
  }
}
//-----------------------------------------------------------------------------
HomotopyODE::~HomotopyODE()
{
  if ( tmp ) delete [] tmp;
}
//-----------------------------------------------------------------------------
complex HomotopyODE::z0(unsigned int i)
{
  const real pp = static_cast<real>(homotopy.degree(i));
  const real mm = static_cast<real>(homotopy.mi[i]);
  const complex c = homotopy.ci[i];

  // Pick root number m of equation z_i^(p + 1) = c_i
  real r = std::pow(std::abs(c), 1.0/(pp + 1.0));
  real a = std::arg(c) / (pp + 1.0);
  complex z = std::polar(r, a + mm/(pp + 1.0)*2.0*DOLFIN_PI);
    
  cout << "Starting point: z = " << z << endl;
  
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

  // Negate F
  for (uint i = 0; i < n; i++)
    y[i] = -y[i];
  
  // Add G(z) = z_i^(p_i + 1) - c_i
  for (uint i = 0; i < n; i++)
    y[i] += Gi(z[i], i);
}
//-----------------------------------------------------------------------------
void HomotopyODE::M(const complex x[], complex y[], const complex z[], real t)
{
  // Need to compute ((1 - t) G' + t F')*x

  // Call user-supplied function to compute F'*x
  homotopy.JF(z, x, y);

  // Multiply by t
  for (uint i = 0; i < n; i++)
    y[i] *= t;

  // Add (1-t) G'*x
  for (uint i = 0; i < n; i++)
  {
    const uint p = homotopy.degree(i);
    const complex zi = z[i];
    complex tmp = static_cast<complex>(p + 1);
    for (uint j = 0; j < p; j++)
      tmp *= zi;
    y[i] += (1.0 - t)* tmp * x[i];
  }
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

  // Negate F'*x
  for (uint i = 0; i < n; i++)
    y[i] = -y[i];

  // Add G'*x
  for (uint i = 0; i < n; i++)
  {
    const uint p = homotopy.degree(i);
    const complex zi = z[i];
    complex tmp = static_cast<complex>(p + 1);
    for (uint j = 0; j < p ; j++)
      tmp *= zi;
    y[i] += tmp * x[i];
  }
}
//-----------------------------------------------------------------------------
bool HomotopyODE::update(const complex z[], real t, bool end)
{
  // Check if we should monitor the homotopy
  if ( tmp )
    monitor(z, t);
  
  // Check convergence of (1 - t)*G(z)
  const real epsilon = 0.5;
  for (uint i = 0; i < n; i++)
  {
    real r = std::abs(pow((1.0 - t), 1.0 - epsilon) * Gi(z[i], i));
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
complex HomotopyODE::Gi(complex zi, uint i) const
{
  // Compute G_i(z_i) = z_i^(p_i + 1) - c_i
  const uint p = homotopy.degree(i);
  const complex ci = homotopy.ci[i];
  complex tmp = zi;
  for (uint j = 0; j < p; j++)
    tmp *= zi;
  
  return tmp - ci;
}
//-----------------------------------------------------------------------------
void HomotopyODE::monitor(const complex z[], real t)
{
  dolfin_assert(tmp);

  // Monitor homotopy H(z, t) = t*F(z) + (1 - t)*G(z)

  // Evaluate F and compute norm of F
  homotopy.F(z, tmp);
  real Fnorm = 0.0;
  for (uint i = 0; i < n; i++)
    Fnorm = std::max(Fnorm, std::abs(tmp[i]));

  // Multiply by t
  for (uint i = 0; i < n; i++)
    tmp[i] *= t;

  // Add (1 - t)*G(z) and compute norm of G
  real Gnorm = 0.0;
  for (uint i = 0; i < n; i++)
  {
    complex G = Gi(z[i], i);
    Gnorm = std::max(Gnorm, std::abs(G));
    tmp[i] += (1.0 - t)*G;
  }
  
  // Compute norm of H
  real Hnorm = 0.0;
  for (uint i = 0; i < n; i++)
    Hnorm = std::max(Hnorm, std::abs(tmp[i]));

  dolfin_info("Homotopy: F = %e G = %e H = %e", Fnorm, Gnorm, Hnorm);
}
//-----------------------------------------------------------------------------
