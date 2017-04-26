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
// First added:  2012-11-09
// Last changed:

#ifndef __TRILINOS_PRECONDITIONER_H
#define __TRILINOS_PRECONDITIONER_H

#ifdef HAS_TRILINOS

#include <memory>

namespace dolfin
{

  class BelosKrylovSolver;
  class TpetraMatrix;

  /// This class provides a common base for Trilinos preconditioners.

  class TrilinosPreconditioner
  {
  public:

    /// Constructor
    TrilinosPreconditioner()
    {}

    /// Destructor
    ~TrilinosPreconditioner()
    {}

    /// Set this preconditioner on a solver
    virtual void set(BelosKrylovSolver& solver) = 0;

    /// Initialise this preconditioner with the operator P
    virtual void init(std::shared_ptr<const TpetraMatrix> P) = 0;

  };
}

#endif

#endif
