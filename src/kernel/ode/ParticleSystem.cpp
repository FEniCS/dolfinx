// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Vector.h>
#include <dolfin/ParticleSystem.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
ParticleSystem::ParticleSystem(unsigned int n, unsigned int dim) : 
  ODE(2*dim*n), n(n), dim(dim)
{
  // Check dimension
  if ( dim == 0 )
    dolfin_error("Dimension must be at least 1 for a particle system.");
  if ( dim > 3 )
    dolfin_error("Maximum allowed dimension is 3 for a particle system.");
  
  // Compute offset
  offset = dim*n;
  
  // Clear pointer to solution
  u = 0;
}
//-----------------------------------------------------------------------------
ParticleSystem::~ParticleSystem()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real ParticleSystem::x0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::y0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::z0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::vx0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::vy0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::vz0(unsigned int i)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::Fx(unsigned int i, real t)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::Fy(unsigned int i, real t)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::Fz(unsigned int i, real t)
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::mass(unsigned int i, real t)
{
  return 1.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::u0(unsigned int i)
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
real ParticleSystem::f(const Vector& u, real t, unsigned int i)
{
  // Return velocity
  if ( i < offset )
    return u(offset + i);

  // Subtract offset
  i -= offset;

  // Save pointer to solution vector
  this->u = &u;

  // Compute force
  switch (i % dim) {
  case 0:
    return Fx(i /dim, t);
    break;
  case 1:
    return Fy(i/dim, t);
    break;
  case 2:
    return Fz(i/dim, t);
    break;
  default:
    dolfin_error("Illegal dimension.");
  }
  
  return 0.0;
}
//-----------------------------------------------------------------------------
real ParticleSystem::x(unsigned int i)
{
  dolfin_assert(u);
  return (*u)(dim*i);
}
//-----------------------------------------------------------------------------
real ParticleSystem::y(unsigned int i)
{
  dolfin_assert(u);
  return (*u)(dim*i + 1);
}
//-----------------------------------------------------------------------------
real ParticleSystem::z(unsigned int i)
{
  dolfin_assert(u);
  return (*u)(dim*i + 2);
}
//-----------------------------------------------------------------------------
real ParticleSystem::vx(unsigned int i)
{
  dolfin_assert(u);
  return (*u)(offset + dim*i);
}
//-----------------------------------------------------------------------------
real ParticleSystem::vy(unsigned int i)
{
  dolfin_assert(u);
  return (*u)(offset + dim*i + 1);
}
//-----------------------------------------------------------------------------
real ParticleSystem::vz(unsigned int i)
{
  dolfin_assert(u);
  return (*u)(offset + dim*i + 2);
}
//-----------------------------------------------------------------------------
