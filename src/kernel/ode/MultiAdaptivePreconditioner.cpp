// Copyright (C) 2005-2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2005-01-27
// Last changed: 2006-07-07

#include <dolfin/uBlasVector.h>
#include <dolfin/Method.h>
#include <dolfin/MultiAdaptiveTimeSlab.h>
#include <dolfin/MultiAdaptivePreconditioner.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
MultiAdaptivePreconditioner::MultiAdaptivePreconditioner
(MultiAdaptiveTimeSlab& timeslab, const Method& method) : ts(timeslab), method(method)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
MultiAdaptivePreconditioner::~MultiAdaptivePreconditioner()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
void MultiAdaptivePreconditioner::solve(uBlasVector& x,
					const uBlasVector& b) const
{
  // Reset dof
  uint j = 0;

  // Iterate over all elements
  for (uint e = 0; e < ts.ne; e++)
  {
    // Get initial value for element
    const int ep = ts.ee[e];
    const real x0 = ( ep != -1 ? x(ep*method.nsize() + method.nsize() - 1) : 0.0 );

    // Propagate value on element
    for (uint n = 0; n < method.nsize(); n++)
      x(j + n) = x0 + b(j + n);

    // Update dof
    j += method.nsize();
  }
}
//-----------------------------------------------------------------------------
