// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/ODE.h>
#include <dolfin/NewArray.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewMethod.h>
#include <dolfin/NewMethod.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptiveJacobian.h>
#include <dolfin/MultiAdaptivePreconditioner.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptivePreconditioner::MultiAdaptivePreconditioner
(const MultiAdaptiveJacobian& A) : A(A), ts(A.ts), ode(A.ode), method(A.method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MultiAdaptivePreconditioner::~MultiAdaptivePreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MultiAdaptivePreconditioner::solve(NewVector& x, const NewVector& b) const
{
  // Reset dof
  uint j = 0;

  // Reset current sub slab
  int s = -1;

  // Reset elast
  ts.elast = -1;

  // Iterate over all elements
  for (uint e = 0; e < ts.ne; e++)
  {
    // Cover all elements in current sub slab
    s = ts.cover(s, e);

    // Get element data
    const uint i = ts.ei[e];
    const real a = ts.sa[s];
    const real b = ts.sb[s];
    const real k = b - a;
    
    // Get initial value for element
    const int ep = ts.ee[e];
    const real x0 = ( ep != -1 ? ts.jx[ep*method.nsize() + method.nsize() - 1] : ts.u0[i] );
    
    // Iterate over dependencies and sum contributions
    real sum = 0.0;
    const NewArray<uint>& deps = ode.dependencies[i];
    for (uint pos = 0; pos < deps.size(); pos++)
    {
      // Get derivative
      const real dfdu = A.Jvalues[A.Jindices[i] + pos];

      sum += x0 + k*dfdu;


    }

    // Update dof
    j += method.nsize();
  }

  x = b;
}
//-----------------------------------------------------------------------------
