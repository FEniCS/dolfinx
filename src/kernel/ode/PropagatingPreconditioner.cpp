// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewArray.h>
#include <dolfin/NewVector.h>
#include <dolfin/NewMethod.h>
#include <dolfin/NewJacobianMatrix.h>
#include <dolfin/ODE.h>
#include <dolfin/NewTimeSlab.h>
#include <dolfin/NewMethod.h>
#include <dolfin/PropagatingPreconditioner.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
PropagatingPreconditioner::PropagatingPreconditioner(const NewJacobianMatrix& A)
  : A(A), ode(A.ode), ts(A.ts), method(A.method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
PropagatingPreconditioner::~PropagatingPreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void PropagatingPreconditioner::solve(NewVector& x, const NewVector& b) const
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
