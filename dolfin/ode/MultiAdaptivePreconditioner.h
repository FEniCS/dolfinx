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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with DOLFIN. If not, see <http://www.gnu.org/licenses/>.
//
// First added:  2005-01-27
// Last changed: 2006-07-07

#ifndef __MULTI_ADAPTIVE_PRECONDITIONER_H
#define __MULTI_ADAPTIVE_PRECONDITIONER_H

#include <dolfin/la/uBLASPreconditioner.h>

namespace dolfin
{
  class ODE;
  class Method;
  class MultiAdaptiveTimeSlab;

  /// This class implements a preconditioner for the Newton system to
  /// be solved on a multi-adaptive time slab. The preconditioner just
  /// does simple forward propagation of values on internal elements
  /// of the time slab (without so much as looking at the Jacobian).

  class MultiAdaptivePreconditioner : public uBLASPreconditioner
  {
  public:

    /// Constructor
    MultiAdaptivePreconditioner(MultiAdaptiveTimeSlab& timeslab, const Method& method);

    /// Destructor
    ~MultiAdaptivePreconditioner();

    /// Solve linear system approximately for given right-hand side b
    void solve(uBLASVector& x, const uBLASVector& b) const;

  private:

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // Method, mcG(q) or mdG(q)
    const Method& method;

  };

}

#endif
