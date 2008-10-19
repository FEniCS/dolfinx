// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2005-05-02
// Last changed: 2008-04-22

#include <dolfin/common/constants.h>
#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/math/Lagrange.h>
#include <dolfin/quadrature/RadauQuadrature.h>
#include <dolfin/la/uBLASVector.h>
#include <dolfin/la/uBLASDenseMatrix.h>
#include <dolfin/ode/ODE.h>
#include <dolfin/ode/SORSolver.h>
#include "dGqMethod.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
dGqMethod::dGqMethod(unsigned int q) : Method(q, q + 1, q + 1)
{
  message("Initializing discontinuous Galerkin method dG(%d).", q);

  init();

  _type = Method::dG;

  p = 2*q + 1;
}
//-----------------------------------------------------------------------------
real dGqMethod::ueval(real x0, real values[], real tau) const
{
  // Note: x0 is not used, maybe this can be done differently

  real sum = 0.0;
  for (unsigned int i = 0; i < nn; i++)
    sum += values[i] * trial->eval(i, tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
real dGqMethod::residual(real x0, real values[], real f, real k) const
{
  // FIXME: Include jump term in residual
  real sum = 0.0;
  for (uint i = 0; i < nn; i++)
    sum += values[i] * derivatives[i];

  return sum / k - f;
}
//-----------------------------------------------------------------------------
real dGqMethod::timestep(real r, real tol, real k0, real kmax) const
{
  // FIXME: Missing stability factor and interpolation constant
  // FIXME: Missing jump term
  
  if ( abs(r) < ODE::epsilon() )
    return kmax;

  //return pow(tol / fabs(r), 1.0 / static_cast<real>(q+1));

  const real qq = static_cast<real>(q);
  return pow(tol * pow(k0, q) / abs(r), 1.0 / (2.0*qq + 1.0));
}
//-----------------------------------------------------------------------------
real dGqMethod::error(real k, real r) const
{
  // FIXME: Missing jump term and interpolation constant
  return pow(k, static_cast<real>(q + 1)) * abs(r);
}
//-----------------------------------------------------------------------------
void dGqMethod::disp() const
{
  message("Data for the dG(%d) method", q);
  message("==========================");
  message("");

  message("Radau quadrature points and weights on [0,1]:");
  message("");
  message(" i   points                   weights");
  message("----------------------------------------------------");
  
  for (unsigned int i = 0; i < nq; i++)
    message("%2d   %.15e   %.15e", i, to_double(qpoints[i]), to_double(qweights[i]));
  message("");

  for (unsigned int i = 0; i < nn; i++)
  {
    message("");
    message("dG(%d) weights for degree of freedom %d:", q, i);
    message("");
    message(" i   weights");
    message("---------------------------");
    for (unsigned int j = 0; j < nq; j++)
      message("%2d   %.15e", j, to_double(nweights[i][j]));
  }
  message("");

  message("dG(%d) weights in matrix format:", q);
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
void dGqMethod::computeQuadrature()
{
  // Use Radau quadrature
  RadauQuadrature quadrature(nq);

  // Get points, rescale from [-1,1] to [0,1], and reverse the points
  for (unsigned int i = 0; i < nq; i++)
  {
    qpoints[i] = 1.0 - (quadrature.point(nq - 1 - i) + 1.0) / 2.0;
    npoints[i] = qpoints[i];
  }

  // Get points, rescale from [-1,1] to [0,1], and reverse the points
  for (unsigned int i = 0; i < nq; i++)
    qweights[i] = 0.5 * quadrature.weight(nq - 1 - i);
}
//-----------------------------------------------------------------------------
void dGqMethod::computeBasis()
{
  dolfin_assert(!trial);
  dolfin_assert(!test);

  // Compute Lagrange basis for trial space
  trial = new Lagrange(q);
  for (unsigned int i = 0; i < nq; i++)
    trial->set(i, qpoints[i]);

  // Compute Lagrange basis for test space
  test = new Lagrange(q);
  for (unsigned int i = 0; i < nq; i++)
    test->set(i, qpoints[i]);
}
//-----------------------------------------------------------------------------
void dGqMethod::computeWeights()
{
  uBLASDenseMatrix A(nn, nn);
  ublas_dense_matrix& _A = A.mat();
  
  real A_real[nn*nn];
  real_zero(nn*nn, A_real);

  // Compute matrix coefficients
  for (unsigned int i = 0; i < nn; i++)
  {
    for (unsigned int j = 0; j < nn; j++)
    {
      // Use Radau quadrature which is exact for the order we need, 2q
      real integral = 0.0;
      for (unsigned int k = 0; k < nq; k++)
      {
        real x = qpoints[k];
        integral += qweights[k] * trial->ddx(j, x) * test->eval(i, x);
      }     
      
      A_real[i*q+j] = integral + trial->eval(j, 0.0) * test->eval(i, 0.0);
      _A(i, j) = to_double(A_real[i*q+j]);
    }
  }

  uBLASVector b(nn);
  uBLASVector w(nn);
  ublas_vector& _b = b.vec();
  ublas_vector& _w = w.vec();

  real b_real[q];

  // Compute nodal weights for each degree of freedom (loop over points)
  for (unsigned int i = 0; i < nq; i++)
  {
    // Get nodal point
    real x = qpoints[i];
    
    // Evaluate test functions at current nodal point
    for (unsigned int j = 0; j < nn; j++)
    {
      b_real[j] = test->eval(j, x);
      _b[j] = to_double(b_real[j]);
    }

    // Solve for the weight functions at the nodal point
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
    
    // Allocate memory for the preconditioned system
    real Ainv_A[q*q];
    real Ainv_b[q];
    
    SORSolver::precondition(q, A_inv, A_real, b_real, Ainv_A, Ainv_b);
    SORSolver::SOR(q, Ainv_A, w_real, Ainv_b, ODE::epsilon());
    
    for (uint j = 0; j < nn; ++j)
      nweights[j][i] = qweights[i] * w_real[j];
    
#endif

  }
}
//-----------------------------------------------------------------------------
