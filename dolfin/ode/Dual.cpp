// Copyright (C) 2003-2005 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Benjamin Kehlet
//
// First added:  2003-11-28
// Last changed: 2008-06-18

#include <dolfin/log/dolfin_log.h>
#include <dolfin/math/dolfin_math.h>
#include "Dual.h"

using namespace dolfin;

//------------------------------------------------------------------------
Dual::Dual(ODE& primal, ODESolution& u) 
  : ODE(primal.size(), primal.endtime()),
    primal(primal), u(u)
{

  // inherit parameters from primal problem
  set("parent", primal);
}
//------------------------------------------------------------------------
Dual::~Dual()
{
  // Do nothing
}
//------------------------------------------------------------------------
void Dual::u0(uBLASVector& y)
{
  // TODO
  // Should be able to use different choices of initial data to the dual
  //return 1.0 / sqrt(static_cast<real>(N));

  y[0] = 1.0;
  for (uint i = 1; i < size(); ++i)  y[i] = 0.0;
  
}
//------------------------------------------------------------------------
void Dual::f(const uBLASVector& phi, real t, uBLASVector& y)
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
  if (tmp.size() != size()) 
    tmp.init(size());
  
  u.eval(endtime()-t, tmp);
  primal.JT(phi, y, tmp, endtime()-t);
  
}
//------------------------------------------------------------------------
real Dual::time(real t) const
{
  return endtime() - t;
}
//-----------------------------------------------------------------------------
