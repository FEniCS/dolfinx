// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Lagrange.h>
#include <dolfin/Method.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Method::Method(unsigned int q)
{
  this->q = q;
  n = q + 1;

  // Allocate points
  points  = new real[n];
  for (unsigned int i = 0; i < n; i++)
    points[i] = 0.0;

  // Allocate weights
  weights = new (real *)[n];
  for (unsigned int i = 0; i < n; i++) {
    weights[i] = new real[n];
    for (unsigned int j = 0; j < n; j++)
      weights[i][j] = 0.0;
  }
  
  // Allocate quadrature weights
  qweights = new real[n];
  for (unsigned int i = 0; i < n; i++)
    qweights[i] = 0.0;

  // Allocate derivatives
  derivatives = new real[n];
  for (unsigned int i = 0; i < n; i++)
    derivatives[i] = 0.0;

  trial = 0;
  test = 0;
}
//-----------------------------------------------------------------------------
Method::~Method()
{
  // Clear points
  if ( points )
    delete [] points;
  points = 0;
  
  // Clear weights
  if ( weights ) {
    for (unsigned int i = 0; i < n; i++)
      delete [] weights[i];
    delete [] weights;
  }
  weights = 0;

  // Clear quadrature weights
  if ( qweights )
    delete [] qweights;
  qweights = 0;
  
  // Clear derivatives
  if ( derivatives )
    delete [] derivatives;
  derivatives = 0;

  // Clear trial basis
  if ( trial )
    delete trial;
  trial = 0;
  
  // Clear test basis
  if ( test )
    delete test;
  test = 0;
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
void Method::computeDerivatives()
{
  for (unsigned int i = 0; i < n; i++)
    derivatives[i] = derivative(i, 1.0);
}
//-----------------------------------------------------------------------------
