// Copyright (C) 2008 Benjamin Kehlet
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
//
// Modified by Garth N. Wells, 2010.
// Modified by Anders Logg, 2011.
//
// First added:  2009-01-25
// Last changed: 2011-01-24

#include "real.h"
#include "constants.h"
#include <dolfin/log/LogStream.h>

using namespace dolfin;

real dolfin::_real_epsilon = DOLFIN_EPS;

// Used to give warning if extended precision is not initialized
bool dolfin::_real_initialized = false;

//-----------------------------------------------------------------------------
void dolfin::set_precision(uint decimal_prec)
{
  #ifdef HAS_GMP
  if (_real_initialized)
    error("dolfin::set_precision called twice.");

  // Compute the number of bits needed
  // set the GMP default precision
  mpf_set_default_prec(static_cast<uint>(decimal_prec*std::log(10)/std::log(2)));

  // Compute epsilon
  real eps = 0.1;
  const real one = real("1.0");
  while (eps + one != one)
    eps /= 2;

  eps *= 2;

  _real_epsilon = eps;

  const int d = mpf_get_default_prec();
  // Display number of digits
  cout << "Using " << d << " bits pr number, epsilon = " << real_epsilon() << endl;
  _real_initialized = true;

  #else
  warning("Can't change floating-point precision when using type double.");
  #endif
}
//-----------------------------------------------------------------------------
int dolfin::real_decimal_prec()
{
  #ifndef HAS_GMP
  return 15;
  #else
  int prec;
  real_frexp(&prec, real_epsilon());
  return static_cast<int>( std::abs(prec*std::log(2)/std::log(10)) );
  #endif
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

    next[A] = (prev[A] + prev[B])/2;
    next[B] = real_sqrt(prev[A]*prev[B]);

    next[T]= prev[T] - P*(prev[A] - next[A])*(prev[A] - next[A]);
    P *= 2;

    pi_next = (next[A] + next[B])*(next[A] + next[B])/(4*next[T]);

  } while (real_abs(pi_next - pi_prev) > 10*real_epsilon());

  return pi_next;
  #endif
}
//-----------------------------------------------------------------------------
double dolfin::real_frexp(int* exp, const real& x)
{
  #ifdef HAS_GMP
  long tmp_long = *exp;
  const double tmp_double = mpf_get_d_2exp(&tmp_long, x.get_mpf_t());
  *exp = static_cast<int>(tmp_long);
  return tmp_double;
  #else
  return frexp(x, exp);
  #endif
}
