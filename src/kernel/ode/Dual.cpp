// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/dolfin_math.h>
#include <dolfin/Function.h>
#include <dolfin/Dual.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Dual::Dual(ODE& primal, Function& u) : ODE(primal.size()), rhs(primal, u)
{
  // Set sparsity to transpose of sparsity for the primal
  sparsity.transp(primal.sparsity);

  // Set end time
  T = primal.endtime();
}
//-----------------------------------------------------------------------------
Dual::~Dual()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
real Dual::u0(unsigned int i)
{
  return 1.0 / sqrt(static_cast<real>(N));
}
//-----------------------------------------------------------------------------
real Dual::f(const Vector& phi, real t, unsigned int i)
{
  // FIXME: Here we can do some optimization. Since we compute the sum
  // FIXME: over all dual dependencies of i we will do the update of
  // FIXME: buffer values many times. Since there is probably some overlap
  // FIXME: we could precompute a new sparsity pattern taking all these
  // FIXME: dependencies into account and then updating the buffer values
  // FIXME: outside the sum.

  real sum = 0.0;  
  
  if ( sparsity.sparse() )
  {
    const NewArray<unsigned int>& row(sparsity.row(i));
    for (unsigned int pos = 0; pos < row.size(); pos++)
      sum += rhs.dfdu(row[pos], i, T - t) * phi(row[pos]);
  }
  else
  {
    for (unsigned int j = 0; j < N; j++)
      sum += rhs.dfdu(j, i, T - t) * phi(j);
  }

  return sum;
}
//-----------------------------------------------------------------------------
