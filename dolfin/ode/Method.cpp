// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2003
// Last changed: 2006-03-27

#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include <dolfin/math/Lagrange.h>
#include "Method.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Method::Method(unsigned int q, unsigned int nq, unsigned int nn)
{
  dolfin_assert(nq > 0);
  dolfin_assert(nn > 0);

  this->q = q;
  this->nq = nq;
  this->nn = nn;

  // Allocate quadrature points
  qpoints  = new double[nq];
  for (unsigned int i = 0; i < nq; i++)
    qpoints[i] = 0.0;

  // Allocate nodal points
  npoints  = new double[nn];
  for (unsigned int i = 0; i < nn; i++)
    npoints[i] = 0.0;

  // Allocate quadrature weights
  qweights = new double[nq];
  for (unsigned int i = 0; i < nq; i++)
    qweights[i] = 0.0;

  // Allocate weights
  nweights = new double*[nn];
  for (unsigned int i = 0; i < nn; i++)
  {
    nweights[i] = new double[nq];
    for (unsigned int j = 0; j < nq; j++)
      nweights[i][j] = 0.0;
  }

  // Allocate derivatives
  derivatives = new double[nq];
  for (unsigned int i = 0; i < nq; i++)
    derivatives[i] = 0.0;

  trial = 0;
  test = 0;

  _type = none;
}
//-----------------------------------------------------------------------------
Method::~Method()
{
  if ( qpoints ) delete [] qpoints;
  if ( npoints ) delete [] npoints;
  if ( qweights ) delete [] qweights;

  if ( nweights )
  {
    for (unsigned int i = 0; i < nn; i++)
      delete [] nweights[i];
    delete [] nweights;
  }

  if ( derivatives ) delete [] derivatives;

  if ( trial ) delete trial;
  if ( test ) delete test;
}
//-----------------------------------------------------------------------------
void Method::init()
{
  computeQuadrature();
  computeBasis();
  computeWeights();
  computeDerivatives();
}
//-----------------------------------------------------------------------------
void Method::update(double x0, double f[], double k, double values[]) const
{
  // Update values
  for (uint i = 0; i < nn; i++)
  {
    double sum = 0.0;
    for (uint j = 0; j < nq; j++)
      sum += nweights[i][j] * f[j];
    values[i] = x0 + k*sum;
  }
}
//-----------------------------------------------------------------------------
void Method::update(double x0, double f[], double k, double values[], double alpha) const
{
  // Update values
  for (uint i = 0; i < nn; i++)
  {
    double sum = 0.0;
    for (uint j = 0; j < nq; j++)
      sum += nweights[i][j] * f[j];
    values[i] += alpha*(x0 + k*sum - values[i]);
  }
}
//-----------------------------------------------------------------------------
void Method::computeDerivatives()
{
  for (unsigned int i = 0; i < nq; i++)
    derivatives[i] = trial->ddx(i, 1.0);
}
//-----------------------------------------------------------------------------
