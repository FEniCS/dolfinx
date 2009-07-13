// Copyright (C) 2008 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-01-25
// Last changed: 2009-03-27

#include "real.h"
#include "constants.h"
#include <dolfin/log/LogStream.h>

using namespace dolfin;


real dolfin::_real_epsilon = DOLFIN_EPS;
// Used to give warning if extended precision is not initialized
bool dolfin::_real_initialized = false;

//-----------------------------------------------------------------------------
void dolfin::dolfin_set_precision(uint decimal_prec) 
{
#ifdef HAS_GMP
  if (_real_initialized)
    error("dolfin_set_precision called twice");

  // Compute the number of bits needed
  // set the GMP default precision
  mpf_set_default_prec(static_cast<uint>(decimal_prec*std::log(10)/std::log(2)));
  
  // Compute epsilon
  real eps = 0.1;
  real one = real("1.0");
  while (eps + one != one)
  {
    eps /= 2;
  }

  eps *= 2;

  _real_epsilon = eps;
  
  int d = mpf_get_default_prec();
  // Display number of digits
  cout << "Using " << d << " bits pr digit, epsilon = " << real_epsilon() << endl;
  _real_initialized = true;
#else 
    warning("Can't change floating-point precision when using type double.");
#endif

}
//-----------------------------------------------------------------------------
int dolfin::real_decimal_prec() {
#ifndef HAS_GMP
  return 15;
#else 
  int prec;
  double dummy = real_frexp(&prec, real_epsilon());
  dummy++; //avoid compiler warning about unused variable
  return static_cast<int>(std::abs( prec * std::log(2)/std::log(10) ));
#endif
}
//-----------------------------------------------------------------------------
real dolfin::real_sqrt(const real& a)
{
  //Solving x^2 - a = 0 using newtons method
  real x(1.0);
  real prev(0.0);

  int k = 0;

  while (real_abs(x - prev) > real_epsilon())
  {
    prev = x;
    x = prev - (prev*prev - a)/(2*prev);
    ++k;
  }
  
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

    } while (real_abs(pi_next - pi_prev) > 10*real_epsilon());

    //gmp_printf("Pi computed in %d iterations: %.50Fe\n", k, pi_next.get_mpf_t());

    return pi_next;
#endif
}
//-----------------------------------------------------------------------------
double dolfin::real_frexp(int* exp, const real& x)
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
