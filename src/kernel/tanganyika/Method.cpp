// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Method.h>

//-----------------------------------------------------------------------------
Method::Method(int q)
{
  if ( q < 0 )
    dolfin_error("Polynomial order must be non-negative.");

  this->q = q;
  n = q + 1;

  points  = new real[n];
  for (int i = 0; i < n; i++)
    points[i] = 0.0;

  weights = new real[n];
  for (int i = 0; i < n; i++)
    weights[i] = 0.0;

  qweights = new real[n];
  for (int i = 0; i < n; i++)
    qweights[i] = 0.0;

  trial = 0;
  test = 0;
}
//-----------------------------------------------------------------------------
Method::~Method()
{
  if ( points )
    delete [] points;
  points = 0;

  if ( weights )
    delete [] weights;
  weights = 0;

  if ( qweights )
    delete [] qweights;
  qweights = 0;
  
  if ( trial )
    delete trial;
  trial = 0;

  if ( test )
    delete test;
  test = 0;
}
//-----------------------------------------------------------------------------
int Method::size() const
{
  return n;
}
//-----------------------------------------------------------------------------
int Method::degree() const
{
  return q;
}
//-----------------------------------------------------------------------------
real Method::point(int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < n);
  dolfin_assert(points);

  return points[i];
}
//-----------------------------------------------------------------------------
real Method::weight(int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < n);
  dolfin_assert(weights);

  return weights[i];
}
//-----------------------------------------------------------------------------
real Method::qweight(int i) const
{
  dolfin_assert(i >= 0);
  dolfin_assert(i < n);
  dolfin_assert(qweights);

  return qweights[i];
}
//-----------------------------------------------------------------------------
void Method::init()
{
  computeQuadrature();
  computePoints();
  computeBasis();
  computeWeights();
}
//-----------------------------------------------------------------------------
