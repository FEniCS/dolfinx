// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Lagrange.h>
#include <dolfin/RadauQuadrature.h>
#include <dolfin/Vector.h>
#include <dolfin/Matrix.h>
#include <dolfin/NewdGqMethod.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewdGqMethod::NewdGqMethod(unsigned int q) : NewMethod(q, q + 1, q + 1)
{
  init();
}
//-----------------------------------------------------------------------------
real NewdGqMethod::ueval(real x0, real values[], real tau) const
{
  // Note: x0 is not used, maybe this can be done differently

  real sum = 0.0;
  for (unsigned int i = 0; i < nn; i++)
    sum += values[i] * trial->eval(i, tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
real NewdGqMethod::ueval(real x0, real values[], uint i) const
{
  return values[i];
}
//-----------------------------------------------------------------------------
real NewdGqMethod::residual(real x0, real values[], real f, real k) const
{
  // FIXME: Include jump term in residual
  real sum = 0.0;
  for (uint i = 0; i < nn; i++)
    sum += values[i] * derivatives[i];

  return sum / k - f;
}
//-----------------------------------------------------------------------------
real NewdGqMethod::timestep(real r, real tol, real kmax) const
{
  // FIXME: Missing stability factor and interpolation constant
  // FIXME: Missing jump term
  
  if ( fabs(r) < DOLFIN_EPS )
    return kmax;

  return pow(tol / fabs(r), 1.0 / static_cast<real>(q+1));
}
//-----------------------------------------------------------------------------
void NewdGqMethod::disp() const
{
  dolfin_info("Data for the dG(%d) method", q);
  dolfin_info("==========================");
  dolfin_info("");

  dolfin_info("Radau quadrature points and weights on [0,1]:");
  dolfin_info("");
  dolfin_info(" i   points                   weights");
  dolfin_info("----------------------------------------------------");
  
  for (unsigned int i = 0; i < nq; i++)
    dolfin_info("%2d   %.16e   %.16e", i, qpoints[i], qweights[i]);
  dolfin_info("");

  for (unsigned int i = 0; i < nn; i++)
  {
    dolfin_info("");
    dolfin_info("dG(%d) weights for degree of freedom %d:", q, i);
    dolfin_info("");
    dolfin_info(" i   weights");
    dolfin_info("---------------------------");
    for (unsigned int j = 0; j < nq; j++)
      dolfin_info("%2d   %.16e", j, nweights[i][j]);
  }
  dolfin_info("");

  dolfin_info("dG(%d) weights in matrix format:", q);
  if ( q < 10 )
    dolfin_info("-------------------------------");
  else
    dolfin_info("--------------------------------");
  for (unsigned int i = 0; i < nn; i++)
  {
    for (unsigned int j = 0; j < nq; j++)
      cout << nweights[i][j] << " ";
    cout << endl;
  }
}
//-----------------------------------------------------------------------------
void NewdGqMethod::computeQuadrature()
{
  // Use Radau quadrature
  RadauQuadrature quadrature(nq);

  // Get points, rescale from [-1,1] to [0,1], and reverse the points
  for (unsigned int i = 0; i < nq; i++)
  {
    qpoints[i] = 1.0 - (quadrature.point(nq - 1 - i).x + 1.0) / 2.0;
    npoints[i] = qpoints[i];
  }

  // Get points, rescale from [-1,1] to [0,1], and reverse the points
  for (unsigned int i = 0; i < nq; i++)
    qweights[i] = 0.5 * quadrature.weight(nq - 1 - i);
}
//-----------------------------------------------------------------------------
void NewdGqMethod::computeBasis()
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
void NewdGqMethod::computeWeights()
{
  Matrix A(nn, nn, Matrix::dense);
  
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
      
      A[i][j] = integral + trial->eval(j, 0.0) * test->eval(i, 0.0);
    }
  }

  Vector b(nn);
  Vector w(nn);

  // Compute nodal weights for each degree of freedom (loop over points)
  for (unsigned int i = 0; i < nq; i++)
  {
    // Get nodal point
    real x = qpoints[i];
    
    // Evaluate test functions at current nodal point
    for (unsigned int j = 0; j < nn; j++)
      b(j) = test->eval(j, x);
    
    // Solve for the weight functions at the nodal point
    A.hpsolve(w, b);

    // Save weights including quadrature
    for (unsigned int j = 0; j < nn; j++)
      nweights[j][i] = qweights[i] * w(j);
  }
}
//-----------------------------------------------------------------------------
