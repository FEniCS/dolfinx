// Copyright (C) 2011 Benjamin Kehlet
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2011-02-16
// Last changed: 2011-02-16


#include <dolfin/la/uBLASDenseMatrix.h>
#include <cfloat>
#include "HighPrecision.h"


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
    real_solve_mat_with_preconditioning(n, Q, E, P, real_epsilon());
    real_mult(n*n, E, -2);

    for (uint i = 0; i < n; ++i)
      E[i + n*i] -= 1.0;
  }
  else
  {
    real_mat_prod_inplace(n, P, _A);
    real_sub(n*n, Q, P);

    real_solve_mat_with_preconditioning(n, Q, E, P, real_epsilon());
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
void dolfin::real_solve(uint n,
		    const real* A,
		    real* x,
		    const real* b,
		    const real& tol)
{
  real prev[n];

  uint iterations = 0;

  real diff = DBL_MAX; // some big number

  while ( diff > tol )
  {
    if ( iterations > MAX_ITERATIONS )
      error("SOR: System seems not to converge");

    real_set(n, prev, x);

    //SOR_iteration(n, A, b, x, prev);

    // Do SOR iteration
    for (uint i = 0; i < n; ++i)
    {
      x[i] = b[i];

      // j < i
      for (uint j = 0; j < i; ++j)
        x[i] -= (A[i + n*j] * x[j]);

      // j > i
      for (uint j = i+1; j < n; ++j)
        x[i] -= (A[i + n*j]* prev[j]);

      x[i] /= A[i + n*i];
    }

    // Check tolerance
    real_sub(n, prev, x);
    diff = real_max_abs(n, prev);

    ++iterations;
  }
}
//-----------------------------------------------------------------------------
void dolfin::real_solve_precond(uint n,
			    const real* A,
			    real* x,
			    const real* b,
			    const uBLASDenseMatrix& precond,
			    const real& tol)
{
  real A_precond[n*n];
  real b_precond[n];

  //Compute precond*A and precond*b with extended precision
  for (uint i = 0; i < n; ++i)
  {
    b_precond[i] = 0.0;
    for (uint j = 0; j < n; ++j)
    {
      b_precond[i] += precond(i,j)*b[j];

      A_precond[i + n*j] = 0.0;
      for (uint k = 0; k < n; ++k)
        A_precond[i + n*j] += precond(i, k)* A[k + n*j];
    }
  }

  // use precond*b as initial guess
  real_set(n, x, b_precond);

  // solve the system
  real_solve(n, A_precond, x, b_precond, tol);
}
//-----------------------------------------------------------------------------
void dolfin::real_solve_mat_with_preconditioning(uint n,
					     const real* A,
					     real* X,
					     const real* B,
					     const real& tol)
{

  uBLASDenseMatrix preconditioner(n, n);
  ublas_dense_matrix& _prec = preconditioner.mat();

  // Set the preconditioner matrix
  for (uint i=0; i<n; i++)
  {
    for (uint j = 0; j < n; j++)
      _prec(i,j) = to_double(A[i + n*j]);
  }

  preconditioner.invert();

  //solve each row as a Ax=b system
  for (uint i = 0; i < n; ++i)
    real_solve_precond(n, A, &X[i*n], &B[i*n], preconditioner, tol);
}
//-----------------------------------------------------------------------------
