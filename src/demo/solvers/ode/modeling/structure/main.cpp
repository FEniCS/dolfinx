// Copyright (C) 2004 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin.h>

using namespace dolfin;

class Structure : public ParticleSystem
{
public:
  
  Structure(unsigned int p) : ParticleSystem(p*p + (p-1)*(p-1), 2), p(p)
  {
    // Final time
    T = 10.0;
    //T = 4e-4;

    // Distance between large masses
    h = 1.0 / static_cast<real>(p-1);

    // Distance between small and large masses
    hsmall = 0.5 * sqrt(2.0) * h;

    // The small mass
    m = 1e-12;
    
    // Compute displacements for the small masses
    dx.init((p-1)*(p-1));
    dy.init((p-1)*(p-1));
    for (unsigned int i = 0; i < (p-1)*(p-1); i++)
    {
      unsigned int i1 = i % (p - 1);
      unsigned int i2 = i / (p - 1);

      real d = 0.25;

      if ( i1 % 2 == 0 )
	dx(i) = d*h;
      else
	dx(i) = - d*h;

      if ( i2 % 2 == 0 )
	dy(i) = - d*h;
      else
	dy(i) = d*h;
    }

    // Compute sparsity
    sparse();
  }

  real x0(unsigned int i)
  {
    if ( i < p*p )
      return static_cast<real>(i % p) * h;
    else
    {
      return 0.5*h + static_cast<real>((i-p*p) % (p-1)) * h + dx(i - p*p);
    }
  }

  real y0(unsigned int i)
  {
    if ( i < p*p )
      return static_cast<real>(i / p) * h;
    else
      return 0.5*h + static_cast<real>((i-p*p) / (p-1)) * h + dy(i - p*p);
  }
  
  real mass(unsigned int i, real t)
  {
    if ( i < p*p )
      return 100.0;
    else
      return m;
  }

  real Fx(unsigned int i, real t)
  {
    real fx = 0.0;
    if ( i < p*p )
    {
      // Compute force for large mass
      unsigned int i1 = i % p;
      unsigned int i2 = i / p;
      unsigned int j  = p*p + (i2-1)*(p-1) + (i1 - 1);; // Small mass at south-west

      // Add force from the left
      if ( i1 > 0 )
      {
	real r = dist(i, i - 1);
	fx += (r - h) * (x(i-1) - x(i)) / r;
      }
      
      // Add force from the right
      if ( i1 < (p-1) )
      {
	real r = dist(i, i + 1);
	fx += (r - h) * (x(i+1) - x(i)) / r;
      }

      // Add force from below
      if ( i2 > 0 )
      {
	real r = dist(i, i - p);
	fx += (r - h) * (x(i-p) - x(i)) / r;
      }

      // Add force from above
      if ( i2 < (p-1) )
      {
	real r = dist(i, i + p);
	fx += (r - h) * (x(i+p) - x(i)) / r;
      }

      // Add force from south-west
      if ( i1 > 0 && i2 > 0 && (i1 + i2) % 2 == 0)
      {
	real r = dist(i, j);
	fx += (r - hsmall) * (x(j) - x(i)) / r;
      }

      // Add force from south-east
      if ( i1 < (p-1) && i2 > 0 && (i1 + i2) % 2 == 0)
      {
	real r = dist(i, j + 1);
	fx += (r - hsmall) * (x(j + 1) - x(i)) / r;
      }

      // Add force from north-west
      if ( i1 > 0 && i2 < (p-1) && (i1 + i2) % 2 == 0)
      {
	real r = dist(i, j + p - 1);
	fx += (r - hsmall) * (x(j + p - 1) - x(i)) / r;
      }

      // Add force from north-east
      if ( i1 < (p-1) && i2 < (p-1) && (i1 + i2) % 2 == 0)
      {
	real r = dist(i, j + p - 1 + 1);
	fx += (r - hsmall) * (x(j + p - 1 + 1) - x(i)) / r;
      }      
    }
    else
    {
      // Compute force for small mass
      unsigned int i1 = (i - p*p) % (p - 1);
      unsigned int i2 = (i - p*p) / (p - 1);
      unsigned int j  = i2*p + i1; // Large mass at south-west

      // Add force from south-west
      if ( (i1 + i2) % 2 == 0 )
      {
	real r = dist(i, j);
	fx += (r - hsmall) * (x(j) - x(i)) / r;
      }

      // Add force from south-east
      if ( (i1 + i2) % 2 == 1 )
      {
	real r = dist(i, j + 1);
	fx += (r - hsmall) * (x(j + 1) - x(i)) / r;
      }

      // Add force from north-west
      if ( (i1 + i2) % 2 == 1 )
      {
	real r = dist(i, j + p);
	fx += (r - hsmall) * (x(j + p) - x(i)) / r;
      }

      // Add force from north-east
      if ( (i1+ i2) % 2 == 0 )
      {
	real r = dist(i, j + p + 1);
	fx += (r - hsmall) * (x(j + p + 1) - x(i)) / r;
      }
    }
    
    return fx;
  }

  real Fy(unsigned int i, real t)
  {
    real fy = 0.0;
    if ( i < p*p )
    {
      // Compute force for large mass
      unsigned int i1 = i % p;
      unsigned int i2 = i / p;
      unsigned int j  = p*p + (i2-1)*(p-1) + (i1 - 1);; // Small mass at south-west

      // Add force from the left
      if ( i1 > 0 )
      {
	real r = dist(i, i - 1);
	fy += (r - h) * (y(i-1) - y(i)) / r;
      }
      
      // Add force from the right
      if ( i1 < (p-1) )
      {
	real r = dist(i, i + 1);
	fy += (r - h) * (y(i+1) - y(i)) / r;
      }

      // Add force from below
      if ( i2 > 0 )
      {
	real r = dist(i, i - p);
	fy += (r - h) * (y(i-p) - y(i)) / r;
      }

      // Add force from above
      if ( i2 < (p-1) )
      {
	real r = dist(i, i + p);
	fy += (r - h) * (y(i+p) - y(i)) / r;
      }

      // Add force from south-west
      if ( i1 > 0 && i2 > 0 && (i1 + i2) % 2 == 0)
      {
	real r = dist(i, j);
	fy += (r - hsmall) * (y(j) - y(i)) / r;
      }

      // Add force from south-east
      if ( i1 < (p-1) && i2 > 0 && (i1 + i2) % 2 == 0)
      {
	real r = dist(i, j + 1);
	fy += (r - hsmall) * (y(j + 1) - y(i)) / r;
      }

      // Add force from north-west
      if ( i1 > 0 && i2 < (p-1) && (i1 + i2) % 2 == 0)
      {
	real r = dist(i, j + p - 1);
	fy += (r - hsmall) * (y(j + p - 1) - y(i)) / r;
      }

      // Add force from north-east
      if ( i1 < (p-1) && i2 < (p-1) && (i1 + i2) % 2 == 0)
      {
	real r = dist(i, j + p - 1 + 1);
	fy += (r - hsmall) * (y(j + p - 1 + 1) - y(i)) / r;
      }
    }
    else
    {
      // Compute force for small mass
      unsigned int i1 = (i - p*p) % (p - 1);
      unsigned int i2 = (i - p*p) / (p - 1);
      unsigned int j  = i2 * p + i1;

      // Add force from south-west
      if ( (i1 + i2) % 2 == 0 )
      {
	real r = dist(i, j);
	fy += (r - hsmall) * (y(j) - y(i)) / r;
      }

      // Add force from south-east
      if ( (i1 + i2) % 2 == 1 )
      {
	real r = dist(i, j + 1);
	fy += (r - hsmall) * (y(j + 1) - y(i)) / r;
      }

      // Add force from north-west
      if ( (i1 + i2) % 2 == 1 )
      {
	real r = dist(i, j + p);
	fy += (r - hsmall) * (y(j + p) - y(i)) / r;
      }

      // Add force from north-east
      if ( (i1 + i2) % 2 == 0 )
      {
	real r = dist(i, j + p + 1);
	fy += (r - hsmall) * (y(j + p + 1) - y(i)) / r;
      }
    }

    return fy;
  }
  
private:

  unsigned int p;
  real h;
  real hsmall;
  real m;

  Vector dx;
  Vector dy;

};

int main()
{
  dolfin_set("output", "plain text");
  dolfin_set("tolerance", 0.001);
  dolfin_set("initial time step", 0.0000001);
  dolfin_set("solve dual problem", false);
  dolfin_set("number of samples", 500);
  dolfin_set("automatic modeling", true);
  dolfin_set("average length", 0.0001);
  dolfin_set("average samples", 100);
  dolfin_set("element cache size", 128);

  Structure structure(5);
  structure.solve();
  
  return 0;
}
