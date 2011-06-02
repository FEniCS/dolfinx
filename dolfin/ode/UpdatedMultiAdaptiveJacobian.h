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
// Last changed: 2006-08-08

#ifndef __UPDATED_MULTI_ADAPTIVE_JACOBIAN_H
#define __UPDATED_MULTI_ADAPTIVE_JACOBIAN_H

#include "TimeSlabJacobian.h"

namespace dolfin
{
  class MultiAdaptiveNewtonSolver;
  class MultiAdaptiveTimeSlab;

  /// This class represents the Jacobian matrix of the system of
  /// equations defined on a multi-adaptive time slab.

  class UpdatedMultiAdaptiveJacobian : public TimeSlabJacobian
  {
  public:

    /// Constructor
    UpdatedMultiAdaptiveJacobian(MultiAdaptiveNewtonSolver& newton,
			     MultiAdaptiveTimeSlab& timeslab);

    /// Destructor
    ~UpdatedMultiAdaptiveJacobian();

    /// Return number of rows (dim = 0) or columns (dim = 1)
    uint size(uint dim) const;

    /// Compute product y = Ax
    void mult(const uBLASVector& x, uBLASVector& y) const;

    /// (Re-)initialize computation of Jacobian
    void init();

    /// Friends
    friend class MultiAdaptivePreconditioner;

  private:

    // The Newton solver
    MultiAdaptiveNewtonSolver& newton;

    // The time slab
    MultiAdaptiveTimeSlab& ts;

    // Size of increment
    double h;

  };

}

#endif
