// Copyright (C) 2008 Benjamin Kehlet.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2010.
// Modified by Anders Logg, 2011.
//
// First added:  2009-01-25
// Last changed: 2011-01-24

#include "real.h"
#include "constants.h"
#include <dolfin/log/LogStream.h>
#include <dolfin/ode/SORSolver.h>

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
//-----------------------------------------------------------------------------
//  This is a port of the routine padm( A, p) from the software package
//  expokit.
//  See: http://www.maths.uq.edu.au/expokit/
//-----------------------------------------------------------------------------
void dolfin::real_mat_exp(uint n, real* E, const real* A, const uint p)
{
  real _A[n*n];
  real A2[n*n];
  real  P[n*n];
  real  Q[n*n];

  // A is const. Use _A instead
  real_set(n*n, _A, A);

  // Calcuate Pade coefficients  (1-based instead of 0-based as in litterature)
  real c[p+2];
  c[1] = 1.0;
  for (uint k = 1; k <= p; ++k)
    c[k+1] = c[k]*((p + 1.0 - k)/(k*(2*p + 1 - k)));

  real norm = 0.0;

  // Scaling

  // compute infinty norm of A
  for (uint i = 0; i < n; ++i)
  {
    real tmp = 0.0;
    for(uint j = 0; j < n; j++)
      tmp += real_abs(A[i + n*j]);
    norm = real_max(norm, tmp);
  }

  uint s = 0;
  if (norm > 0.5)
  {
    s = std::max(0, static_cast<int>(std::log(to_double(norm)) / std::log(2.0)) + 2);
    real_mult(n*n, _A, 1.0/real_pow(2, s));
  }

  // Horner evaluation of the irreducible fraction
  real_mat_prod(n, A2, _A, _A);
  real_identity(n, Q, c[p + 1]);
  real_identity(n, P, c[p]);

  bool odd = true;
  for( uint k = p - 1; k > 0; --k)
  {
    if (odd)
    {
      //Q = Q*A2
      real_mat_prod_inplace(n, Q, A2);
      // Q += c(k)*I
      for (uint i = 0; i<n; i++) Q[i + n*i] += c[k];
    }
    else
    {
      // P = P*A2
      real_mat_prod_inplace(n, P, A2);

      // P += c(k)*I
      for (uint i = 0; i < n; i++)
        P[i + n*i] += c[k];
    }
    odd = !odd;
  }

  if (odd)
  {
    // Q = Q*A
    real_mat_prod_inplace(n, Q, _A);
    // Q = Q - P
    real_sub(n*n, Q, P);
    // E = -(I + 2*(Q\P));

    // find Q\P
    SORSolver::SOR_mat_with_preconditioning(n, Q, E, P, real_epsilon());
    real_mult(n*n, E, -2);

    for (uint i = 0; i < n; ++i)
      E[i + n*i] -= 1.0;
  }
  else
  {
    real_mat_prod_inplace(n, P, _A);
    real_sub(n*n, Q, P);

    SORSolver::SOR_mat_with_preconditioning(n, Q, E, P, real_epsilon());
    real_mult(n*n, E, 2);
    for (uint i = 0; i < n; ++i)
      E[i + n*i] += 1.0;
  }

  // Squaring
  for(uint i = 0; i < s; ++i)
  {
    //use _A as temporary matrix
    real_set(n*n, _A, E);
    real_mat_prod(n, E, _A, _A);
  }
}

//-----------------------------------------------------------------------------
// Matrix multiplication res = A*B
void dolfin::real_mat_prod(uint n, real* res, const real* A, const real* B)
{
  for (uint i = 0; i < n; ++i)
  {
    for (uint j = 0; j < n; ++j)
    {
      res[i + n*j] = 0.0;
      for (uint k = 0; k < n; ++k)
        res[i+n*j] += A[i + n*k]* B[k + n*j];
    }
  }
}
//-----------------------------------------------------------------------------
// Matrix multiplication A = A * B
void dolfin::real_mat_prod_inplace(uint n, real* A, const real* B)
{
  real tmp[n*n];
  real_set(n*n, tmp, A);
  real_mat_prod(n, A, tmp, B);
}
//-----------------------------------------------------------------------------
// Matrix vector product y = Ax
void dolfin::real_mat_vector_prod(uint n, real* y, const real* A, const real* x)
{
  for (uint i = 0; i < n; ++i)
  {
    y[i] = 0;
    for (uint j = 0; j < n; ++j)
      y[i] += A[i + n*j]*x[j];
  }
}
//-----------------------------------------------------------------------------
// Matrix power A = B^q
void dolfin::real_mat_pow(uint n, real* A, const real* B, uint q)
{
  // TODO : Minimize number of matrix multiplications
  real_identity(n, A);
  for (uint i = 0; i < q; ++i)
    real_mat_prod_inplace(n, A, B);
}
//-----------------------------------------------------------------------------
