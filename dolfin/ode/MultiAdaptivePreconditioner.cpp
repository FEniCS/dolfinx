// Copyright (C) 2005-2006 Anders Logg
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
// First added:  2005-01-27
// Last changed: 2006-07-07

#include <dolfin/la/uBLASVector.h>
#include "Method.h"
#include "MultiAdaptiveTimeSlab.h"
#include "MultiAdaptivePreconditioner.h"

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
void MultiAdaptivePreconditioner::solve(uBLASVector& x,
					const uBLASVector& b) const
{
  // Reset dof
  uint j = 0;

  // Iterate over all elements
  for (uint e = 0; e < ts.ne; e++)
  {
    // Get initial value for element
    const int ep = ts.ee[e];
    const double x0 = ( ep != -1 ? x[ep*method.nsize() + method.nsize() - 1] : 0.0 );

    // Propagate value on element
    for (uint n = 0; n < method.nsize(); n++)
      x[j + n] = x0 + b[j + n];

    // Update dof
    j += method.nsize();
  }
}
//-----------------------------------------------------------------------------
