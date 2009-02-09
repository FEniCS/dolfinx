// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-01-25
// Last changed: 2009-02-09

#include "real.h"
#include "constants.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
real dolfin::real_epsilon() 
{
#ifndef HAS_GMP
  return DOLFIN_EPS;
#else 
  static bool computed = false;
  //probably faster to store eps somewhere else and call real_init from GMPObject
  //compute it (thus avoiding the if-test). But where to store eps?
  static real eps;

  if (!computed)
  {
    //computing precision
    eps = 1.0;
    real one = real("1.0");
    while ( eps + one != one ) 
    {
      eps /= 2;
    }
    
    eps *= 2;
    computed = true;
  }
  
  return eps;
#endif
}
//-----------------------------------------------------------------------------
real dolfin::real_sqrt(real a)
{
  //Solving x^2 - a = 0 using newtons method
  real x(1.0);
  real prev(0.0);

  int k = 0;

  while (abs(x - prev) > real_epsilon())
  {
    prev = x;
    x = prev - (prev*prev - a)/(2*prev);
    ++k;
  }
  
  real test = x*x;
  test = test-a;
  /*
  printf("Computed square root in %d iterations\n", k);
  gmp_printf("sqrt, diff: %.20Fe\n", test.get_mpf_t());
  */
  return x;
}
//-----------------------------------------------------------------------------
real dolfin::real_pi()
{
#ifndef HAS_GMP
    return DOLFIN_PI;
#else

    //Computing pi using the Gauss-Legendre formula
  
    real pi_prev;
    real pi_next;
    
    real prev[3];
    real next[3];
    
    const int A = 0;
    const int B = 1;
    const int T = 2;
    
    next[A] = real("1.0");
    next[B] = 1/real_sqrt(real("2.0"));
    next[T] = real("0.25");
    uint P = 1;
    
    uint k = 0;
    do 
    {
      ++k;
      pi_prev = pi_next;
      real_set(3, prev, next);
      
      next[A] = (prev[A]+prev[B])/2;
      next[B] = real_sqrt(prev[A]*prev[B]);
      
      next[T]= prev[T] - P*(prev[A]-next[A])*(prev[A]-next[A]);
      P *= 2;
      
      pi_next = (next[A]+next[B])*(next[A]+next[B])/(4*next[T]);
    
    } while (abs(pi_next - pi_prev) > 10*real_epsilon());

    //gmp_printf("Pi computed in %d iterations: %.50Fe\n", k, pi_next.get_mpf_t());  
    
    return pi_next;
#endif
}
//-----------------------------------------------------------------------------
