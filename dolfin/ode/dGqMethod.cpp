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
double dGqMethod::ueval(double x0, double values[], double tau) const
{
  // Note: x0 is not used, maybe this can be done differently

  double sum = 0.0;
  for (unsigned int i = 0; i < nn; i++)
    sum += values[i] * trial->eval(i, tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
double dGqMethod::ueval(double x0, uBLASVector& values, uint offset, double tau) const
{
  // Note: x0 is not used, maybe this can be done differently

  double sum = 0.0;
  for (unsigned int i = 0; i < nn; i++)
    sum += values[offset + i] * trial->eval(i, tau);
  
  return sum;
}
//-----------------------------------------------------------------------------
double dGqMethod::residual(double x0, double values[], double f, double k) const
{
  // FIXME: Include jump term in residual
  double sum = 0.0;
  for (uint i = 0; i < nn; i++)
    sum += values[i] * derivatives[i];

  return sum / k - f;
}
//-----------------------------------------------------------------------------
double dGqMethod::residual(double x0, uBLASVector& values, uint offset, double f, double k) const
{
  // FIXME: Include jump term in residual
  double sum = 0.0;
  for (uint i = 0; i < nn; i++)
    sum += values[offset + i] * derivatives[i];

  return sum / k - f;
}
//-----------------------------------------------------------------------------
double dGqMethod::timestep(double r, double tol, double k0, double kmax) const
{
  // FIXME: Missing stability factor and interpolation constant
  // FIXME: Missing jump term
  
  if ( fabs(r) < DOLFIN_EPS )
    return kmax;

  //return pow(tol / fabs(r), 1.0 / static_cast<double>(q+1));

  const double qq = static_cast<double>(q);
  return pow(tol * pow(k0, qq) / fabs(r), 1.0 / (2.0*qq + 1.0));
}
//-----------------------------------------------------------------------------
double dGqMethod::error(double k, double r) const
{
  // FIXME: Missing jump term and interpolation constant
  return pow(k, static_cast<double>(q + 1)) * fabs(r);
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
    message("%2d   %.15e   %.15e", i, qpoints[i], qweights[i]);
  message("");

  for (unsigned int i = 0; i < nn; i++)
  {
    message("");
    message("dG(%d) weights for degree of freedom %d:", q, i);
    message("");
    message(" i   weights");
    message("---------------------------");
    for (unsigned int j = 0; j < nq; j++)
      message("%2d   %.15e", j, nweights[i][j]);
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
  
  // Compute matrix coefficients
  for (unsigned int i = 0; i < nn; i++)
  {
    for (unsigned int j = 0; j < nn; j++)
    {
      // Use Radau quadrature which is exact for the order we need, 2q
      double integral = 0.0;
      for (unsigned int k = 0; k < nq; k++)
      {
        double x = qpoints[k];
        integral += qweights[k] * trial->ddx(j, x) * test->eval(i, x);
      }     
      _A(i, j) = integral + trial->eval(j, 0.0) * test->eval(i, 0.0);
    }
  }

  uBLASVector b(nn);
  uBLASVector w(nn);
  ublas_vector& _b = b.vec();
  ublas_vector& _w = w.vec();

  // Compute nodal weights for each degree of freedom (loop over points)
  for (unsigned int i = 0; i < nq; i++)
  {
    // Get nodal point
    double x = qpoints[i];
    
    // Evaluate test functions at current nodal point
    for (unsigned int j = 0; j < nn; j++)
      _b[j] = test->eval(j, x);

    // Solve for the weight functions at the nodal point
    // FIXME: Do we get high enough precision?
    A.solve(w, b);

    // Save weights including quadrature
    for (unsigned int j = 0; j < nn; j++)
      nweights[j][i] = qweights[i] * _w[j];
  }
}
//-----------------------------------------------------------------------------
