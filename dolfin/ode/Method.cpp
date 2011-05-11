// Copyright (C) 2003-2005 Anders Logg
//
// This file is part of DOLFIN.
//
// DOLFIN is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// DOLFIN is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN.  If not, see <http://www.gnu.org/licenses/>.
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
  assert(nq > 0);
  assert(nn > 0);

  this->q = q;
  this->nq = nq;
  this->nn = nn;

  // Allocate quadrature points
  qpoints  = new real[nq];
  for (unsigned int i = 0; i < nq; i++)
    qpoints[i] = 0.0;

  // Allocate nodal points
  npoints  = new real[nn];
  for (unsigned int i = 0; i < nn; i++)
    npoints[i] = 0.0;

  // Allocate quadrature weights
  qweights = new real[nq];
  for (unsigned int i = 0; i < nq; i++)
    qweights[i] = 0.0;

  // Allocate weights
  nweights = new real*[nn];
  for (unsigned int i = 0; i < nn; i++)
  {
    nweights[i] = new real[nq];
    for (unsigned int j = 0; j < nq; j++)
      nweights[i][j] = 0.0;
  }

  // Allocate derivatives
  derivatives = new real[nq];
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
  compute_quadrature();
  compute_basis();
  compute_weights();
  compute_derivatives();
}
//-----------------------------------------------------------------------------
void Method::update(real x0, real f[], real k, real values[]) const
{
  // Update values
  for (uint i = 0; i < nn; i++)
  {
    real sum = 0.0;
    for (uint j = 0; j < nq; j++)
      sum += nweights[i][j] * f[j];
    values[i] = x0 + k*sum;
  }
}
//-----------------------------------------------------------------------------
void Method::update(real x0, real f[], real k, real values[], real alpha) const
{
  // Update values
  for (uint i = 0; i < nn; i++)
  {
    real sum = 0.0;
    for (uint j = 0; j < nq; j++)
      sum += nweights[i][j] * f[j];
    values[i] += alpha*(x0 + k*sum - values[i]);
  }
}
//-----------------------------------------------------------------------------
void Method::compute_derivatives()
{
  for (unsigned int i = 0; i < nq; i++)
    derivatives[i] = trial->ddx(i, 1.0);
}
//-----------------------------------------------------------------------------
