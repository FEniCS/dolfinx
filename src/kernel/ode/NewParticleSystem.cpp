// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2005.

#include <cmath>
#include <dolfin/dolfin_log.h>
#include <dolfin/NewParticleSystem.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewParticleSystem::NewParticleSystem(unsigned int n, unsigned int dim) : 
  ODE(2*dim*n), n(n), dim(dim), offset(0), u(0)
{
  // Check dimension
  if ( dim == 0 )
    dolfin_error("Dimension must be at least 1 for a particle system.");
  if ( dim > 3 )
    dolfin_error("Maximum allowed dimension is 3 for a particle system.");
  
  // Compute offset
  offset = dim*n;

  dolfin_info("Creating particle system of size %d (%d particles).", N, n);
}
//-----------------------------------------------------------------------------
NewParticleSystem::~NewParticleSystem()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real NewParticleSystem::x0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::y0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::z0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::vx0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::vy0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::vz0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::Fx(unsigned int i, real t)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::Fy(unsigned int i, real t)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::Fz(unsigned int i, real t)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::mass(unsigned int i, real t)
{
  return 1.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::k(unsigned int i)
{
  return default_timestep;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::u0(unsigned int i)
{
  if ( i < offset )
  {
    switch (i % dim) {
    case 0:
      return x0(i/dim);
      break;
    case 1:
      return y0(i/dim);
      break;
    case 2:
      return z0(i/dim);
      break;
    default:
      dolfin_error("Illegal dimension.");
    }
  }
  else
  {
    switch (i % dim) {
    case 0:
      return vx0((i-offset)/dim);
      break;
    case 1:
      return vy0((i-offset)/dim);
      break;
    case 2:
      return vz0((i-offset)/dim);
      break;
    default:
      dolfin_error("Illegal dimension.");
    }
  }  

  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::f(real u[], real t, unsigned int i)
{
  // Return velocity
  if ( i < offset )
    return u[offset + i];

  // Subtract offset
  i -= offset;

  // Save pointer to solution vector
  this->u = u;

  // Compute force
  switch (i % dim) {
  case 0:
    return Fx(i/dim, t) / mass(i/dim, t);
    break;
  case 1:
    return Fy(i/dim, t) / mass(i/dim, t);
    break;
  case 2:
    return Fz(i/dim, t) / mass(i/dim, t);
    break;
  default:
    dolfin_error("Illegal dimension.");
  }
  
  return 0.0;
}
//-----------------------------------------------------------------------------
real NewParticleSystem::timestep(unsigned int i)
{
  if ( i >= offset )
    i -= offset;

  i /= dim;

  return k(i);
}
//-----------------------------------------------------------------------------
real NewParticleSystem::dist(unsigned int i, unsigned int j) const
{
  real dx = x(i) - x(j);
  real r = dx*dx;

  if ( dim > 1 )
  {
    real dy = y(i) - y(j);
    r += dy*dy;
  }

  if ( dim > 2 )
  {
    real dz = z(i) - z(j);
    r += dz*dz;
  }
 
  return sqrt(r);
}
//-----------------------------------------------------------------------------
