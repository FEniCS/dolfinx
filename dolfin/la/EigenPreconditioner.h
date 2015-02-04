// Copyright (C) 2015 Chris Richardson
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
// First added:  2015-02-04

#ifndef __EIGEN_PRECONDITIONER_H
#define __EIGEN_PRECONDITIONER_H

#include <dolfin/log/log.h>

namespace dolfin
{

  class EigenVector;
  class EigenLinearOperator;
  class EigenMatrix;

  /// This class specifies the interface for preconditioners for the
  /// Eigen Krylov solver.

  class EigenPreconditioner
  {
  public:

    /// Constructor
    EigenPreconditioner() {}

    /// Destructor
    virtual ~EigenPreconditioner() {}

    /// Initialise preconditioner (sparse matrix)
    virtual void init(const EigenMatrix& P)
    {
      dolfin_error("EigenPreconditioner",
                   "initialize Eigen preconditioner",
                   "No init() function for preconditioner EigenMatrix");
    }

    /// Initialise preconditioner (virtual matrix)
    virtual void init(const EigenLinearOperator& P)
    {
      dolfin_error("EigenPreconditioner",
                   "initialize Eigen preconditioner",
                   "No init() function for preconditioner EigenLinearOperator");
    }

    /// Solve linear system (M^-1)Ax = y
    virtual void solve(EigenVector& x, const EigenVector& b) const = 0;

  };

}

#endif
