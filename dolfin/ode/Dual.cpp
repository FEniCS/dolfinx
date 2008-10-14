// Copyright (C) 2003-2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet
//
// First added:  2003-11-28
// Last changed: 2008-10-06

#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include "Dual.h"

using namespace dolfin;

//------------------------------------------------------------------------
Dual::Dual(ODE& primal, ODESolution& u) 
  : ODE(primal.size(), primal.endtime()),
    primal(primal), u(u)
{

  // Inherit parameters from primal problem
  set("parent", primal);
}
//------------------------------------------------------------------------
Dual::~Dual()
{
  // Do nothing
}
//------------------------------------------------------------------------
void Dual::u0(real* psi)
{
  // FIXME: Enable different choices of initial data to the dual
  //return 1.0 / sqrt(static_cast<real>(N));

  real_zero(size(), psi);
  psi[0] = 1.0;
}
//------------------------------------------------------------------------
void Dual::f(const real* phi, real t, real* y)
{
  // FIXME: Here we can do some optimization. Since we compute the sum
  // FIXME: over all dual dependencies of i we will do the update of
  // FIXME: buffer values many times. Since there is probably some overlap
  // FIXME: we could precompute a new sparsity pattern taking all these
  // FIXME: dependencies into account and then updating the buffer values
  // FIXME: outside the sum.

  /*
  real sum = 0.0;  
  
  if ( sparsity.sparse() )
  {
    const Array<unsigned int>& row(sparsity.row(i));
    for (unsigned int pos = 0; pos < row.size(); pos++)
      sum += rhs.dfdu(row[pos], i, T - t) * phi(row[pos]);
  }
  else
  {
    for (unsigned int j = 0; j < N; j++)
      sum += rhs.dfdu(j, i, T - t) * phi(j);
  }

  return sum;
  */

  // Initialize temporary array if necessary
  if (!tmp0) tmp0 = new real[size()];
  real_zero(size(), tmp0);

  // Evaluate primal at T - t
  u.eval(endtime() - t, tmp0);

  // Evaluate right-hand side
  primal.JT(phi, y, tmp0, endtime() - t);
}
//------------------------------------------------------------------------
real Dual::time(real t) const
{
  return endtime() - t;
}
//-----------------------------------------------------------------------------
