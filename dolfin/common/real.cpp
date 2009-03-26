// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-01-25
// Last changed: 2009-02-09

#include "real.h"
#include "constants.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
real dolfin::_real_epsilon = DOLFIN_EPS;

void dolfin::real_init() {
#ifndef HAS_GMP
  _real_epsilon = DOLFIN_EPS;
#else 
  //computing precision
  real eps = 0.1;
  real one = real("1.0");
  while ( eps + one != one ) 
  {
    eps /= 2;
  }
    
  eps *= 2;
  
  _real_epsilon = eps;
#endif
}
//-----------------------------------------------------------------------------
int dolfin::real_decimal_prec() {
  int prec;
  double dummy = real_frexp(&prec, real_epsilon());
  dummy++; //avoid compiler warning about unused variable
  return std::abs(static_cast<int>( prec * std::log(2)/std::log(10) ));
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
double dolfin::real_frexp(int* exp, real x)
{
#ifdef HAS_GMP
  long tmp_long = *exp;
  double tmp_double = mpf_get_d_2exp(&tmp_long, x.get_mpf_t());
  *exp = static_cast<int>(tmp_long);
  return tmp_double;
#else
  return frexp(x, exp);
#endif
}
