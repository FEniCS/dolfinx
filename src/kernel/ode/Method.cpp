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
unsigned int Method::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
unsigned int Method::degree() const
{
  return q;
}
//-----------------------------------------------------------------------------
real Method::point(unsigned int i) const
{
  dolfin_assert(i < n);
  dolfin_assert(points);

  return points[i];
}
//-----------------------------------------------------------------------------
real Method::weight(unsigned int i, unsigned int j) const
{
  dolfin_assert(i < n);
  dolfin_assert(j < n);
  dolfin_assert(weights);

  return weights[i][j];
}
//-----------------------------------------------------------------------------
real Method::weight(unsigned int i) const
{
  dolfin_assert(i < n);
  dolfin_assert(qweights);

  return qweights[i];
}
//-----------------------------------------------------------------------------
real Method::basis(unsigned int i, real t) const
{
  dolfin_assert(i < n);
  dolfin_assert(trial);

  return trial->eval(i, t);
}
//-----------------------------------------------------------------------------
real Method::derivative(unsigned int i, real t) const
{
  dolfin_assert(i < n);
  dolfin_assert(trial);

  return trial->dx(i, t);
}
//-----------------------------------------------------------------------------
real Method::derivative(unsigned int i) const
{
  dolfin_assert(i < n);
  dolfin_assert(trial);

  return derivatives[i];
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
