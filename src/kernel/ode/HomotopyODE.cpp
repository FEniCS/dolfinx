// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Homotopy.h>
#include <dolfin/HomotopyODE.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
HomotopyODE::HomotopyODE(Homotopy& homotopy, uint n)
  : ComplexODE(n), homotopy(homotopy), n(n), _state(ode)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
HomotopyODE::~HomotopyODE()
{
  // Do nothing
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
void HomotopyODE::feval(const complex z[], real t, complex f[])
{
  // Need to compute f(z) = G(z) - F(z)

  // Call user-supplied function to compute F(z)
  homotopy.F(z, f);

  // FIXMETMP: Remove when working
  return;

  // Just compute F(z) if playing end game
  if ( _state == endgame )
    return;

  // Negate F
  for (uint i = 0; i < n; i++)
    f[i] = -f[i];

  // Add G(z) = z_i^(p_i + 1) - c_i
  for (uint i = 0; i < n; i++)
  {
    const uint p = homotopy.degree(i);
    const complex zi = z[i];
    const complex ci = homotopy.ci[i];
    complex tmp = zi;
    for (uint j = 0; j < p; j++)
      tmp *= zi;
    f[i] += tmp - ci;
  }
}
//-----------------------------------------------------------------------------
void HomotopyODE::M(const complex x[], complex y[], const complex z[], real t)
{
  // Need to compute ((1 - t) G' + t F')*x

  // Call user-supplied function to compute F'*x
  homotopy.JF(z, x, y);

  // FIXMETMP: Remove when working
  return;

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
  // Compute the size of (1 - t)*G(z) and see that it does not diverge;
  // it should converge to zero

  const real tol = 5.0;

  for (uint i = 0; i < n; i++)
  {
    const uint p = homotopy.degree(i);
    const complex zi = z[i];
    const complex ci = homotopy.ci[i];
    complex tmp = zi;
    for (uint j = 0; j < p; j++)
      tmp *= zi;
    real r = std::abs((1 - t) * (tmp - ci));
    cout << "checking: r = " << r << endl;
    
    if ( r > tol )
    {
      dolfin_info("Homotopy path seems to be diverging.");
      return false;
    }

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
  }
  
  return true;
}
//-----------------------------------------------------------------------------
HomotopyODE::State HomotopyODE::state()
{
  return _state;
}
//-----------------------------------------------------------------------------
