// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet 
//
// First added:  2005-05-02
// Last changed: 2008-10-12

#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/math/Lagrange.h>
#include <dolfin/quadrature/LobattoQuadrature.h>
#include <dolfin/la/uBLASVector.h>
#include <dolfin/la/uBLASDenseMatrix.h>
#include <dolfin/ode/ODE.h>
#include <dolfin/ode/SORSolver.h>
#include "cGqMethod.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
cGqMethod::cGqMethod(unsigned int q) : Method(q, q + 1, q)
{
  message("Initializing continuous Galerkin method cG(%d).", q);

  init();

  _type = Method::cG;

  p = 2*q;
}
//-----------------------------------------------------------------------------
real cGqMethod::ueval(real x0, real values[], real tau) const
{
  real sum = x0 * trial->eval(0, tau);
  for (uint i = 0; i < nn; i++)
    sum += values[i] * trial->eval(i + 1, tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
real cGqMethod::residual(real x0, real values[], real f, real k) const
{
  real sum = x0 * derivatives[0];
  for (uint i = 0; i < nn; i++)
    sum += values[i] * derivatives[i + 1];

  return sum / k - f;
}
//-----------------------------------------------------------------------------
real cGqMethod::timestep(real r, real tol, real k0, real kmax) const
{
  // FIXME: Missing stability factor and interpolation constant

  if ( abs(r) < ODE::epsilon() )
    return kmax;

  const real qq = static_cast<real>(q);
  return pow(tol * pow(k0, q) / abs(r), 0.5 / qq);
}
//-----------------------------------------------------------------------------
real cGqMethod::error(real k, real r) const
{
  // FIXME: Missing interpolation constant
  return pow(k, static_cast<real>(q)) * abs(r);
}
//-----------------------------------------------------------------------------
void cGqMethod::disp() const
{
  message("Data for the cG(%d) method", q);
  message("=========================");
  message("");

  message("Lobatto quadrature points and weights on [0,1]:");
  message("");
  message(" i   points                   weights");
  message("----------------------------------------------------");
  
  for (unsigned int i = 0; i < nq; i++)
    message("%2d   %.15e   %.15e", i, to_double(qpoints[i]), to_double(qweights[i]));
  message("");

  for (unsigned int i = 0; i < nn; i++)
  {
    message("");
    message("cG(%d) weights for degree of freedom %d:", q, i);
    message("");
    message(" i   weights");
    message("---------------------------");
    for (unsigned int j = 0; j < nq; j++)
      message("%2d   %.15e", j, to_double(nweights[i][j]));
  }
  message("");
  
  message("cG(%d) weights in matrix format:", q);
  if ( q < 10 )
    message("-------------------------------");
  else
    message("--------------------------------");
  for (unsigned int i = 0; i < nn; i++)
  {
    for (unsigned int j = 0; j < nq; j++)
      cout << nweights[i][j] << " ";
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
void cGqMethod::computeQuadrature()
{
  // Use Lobatto quadrature
  LobattoQuadrature quadrature(nq);

  // Get quadrature points and rescale from [-1,1] to [0,1]
  for (unsigned int i = 0; i < nq; i++)
    qpoints[i] = (quadrature.point(i) + 1.0) / 2.0;

  // Get nodal points and rescale from [-1,1] to [0,1]
  for (unsigned int i = 0; i < nn; i++)
    npoints[i] = (quadrature.point(i + 1) + 1.0) / 2.0;

  // Get quadrature weights and rescale from [-1,1] to [0,1]
  for (unsigned int i = 0; i < nq; i++)
    qweights[i] = 0.5 * quadrature.weight(i);
}
//-----------------------------------------------------------------------------
void cGqMethod::computeBasis()
{
  dolfin_assert(!trial);
  dolfin_assert(!test);

  // Compute Lagrange basis for trial space
  trial = new Lagrange(q);
  for (unsigned int i = 0; i < nq; i++)
    trial->set(i, qpoints[i]);

  // Compute Lagrange basis for test space using the Lobatto points for q - 1
  test = new Lagrange(q - 1);
  if ( q > 1 )
  {
    LobattoQuadrature lobatto(nq - 1);
    for (unsigned int i = 0; i < (nq - 1); i++)
      test->set(i, (lobatto.point(i) + 1.0) / 2.0);
  }
  else
    test->set(0, 1.0);
}
//-----------------------------------------------------------------------------
void cGqMethod::computeWeights()
{
  uBLASDenseMatrix A(q, q);
  ublas_dense_matrix& _A = A.mat();
  
  real A_real[q*q];

  // Compute matrix coefficients
  for (uint i = 0; i < nn; i++)
  {
    for (uint j = 0; j < nn; j++)
    {
      // Use Lobatto quadrature which is exact for the order we need, 2q-1
      real integral = 0.0;
      for (uint k = 0; k < nq; k++)
      {
	      real x = qpoints[k];
	      integral += qweights[k] * trial->ddx(j + 1, x) * test->eval(i, x);
      }

      A_real[i*q+j] = integral;
      _A(i, j) = to_double(integral);

    }
  }


  uBLASVector b(q);
  ublas_vector& _b = b.vec();
  real b_real[q];

  uBLASVector w(q);
  ublas_vector& _w = w.vec();


  // Compute nodal weights for each degree of freedom (loop over points)
  for (uint i = 0; i < nq; i++)
  {
    // Get nodal point
    real x = qpoints[i];
    
    // Evaluate test functions at current nodal point
    for (uint j = 0; j < nn; j++)
    {
      b_real[j] = test->eval(j, x);
      _b[j] = to_double(b_real[j]);
    }
    // Solve for the weight functions at the nodal point
    // FIXME: Do we get high enough precision?
    A.solve(w, b);
    
    #ifndef HAS_GMP
      // Save weights including quadrature
      for (uint j = 0; j < nn; j++)
        nweights[j][i] = qweights[i] * _w[j];

    #else 

      // Use the double precision solution as initial guess for the SOR iterator
      real w_real[q];

      for (uint j = 0; j < q; ++j)
        w_real[j] = _w[j];

      uBLASDenseMatrix A_inv(A);
      A_inv.invert();

      //Allocate memory for holdgin the preconditioned system
      real Ainv_A[q*q];
      real Ainv_b[q];

      SORSolver::precondition(q, A_inv, A_real, b_real, Ainv_A, Ainv_b);
      
      //solve the preconditioned system
      SORSolver::SOR(q, Ainv_A, w_real, Ainv_b, ODE::epsilon());
    
      for (uint j = 0; j < nn; ++j)
        nweights[j][i] = qweights[i] * w_real[j];

    #endif
  }
}
//-----------------------------------------------------------------------------
