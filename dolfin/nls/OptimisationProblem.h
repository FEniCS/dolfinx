// Copyright (C) 2014 Tianyi Li
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
// First added:  2014-06-22
// Last changed: 2014-07-19

#ifndef __OPTIMISATION_PROBLEM_H
#define __OPTIMISATION_PROBLEM_H

#include "NonlinearProblem.h"

namespace dolfin
{

// Forward declarations
class PETScMatrix;
class PETScVector;

/// This is a base class for nonlinear optimisation problems which
/// return the real-valued objective function :math:`f(x)`, its
/// gradient :math:`F(x) = f'(x)` and its Hessian :math:`J(x) =
/// f''(x)`

class OptimisationProblem : public NonlinearProblem
{
public:
  /// Constructor
  OptimisationProblem() {}

  /// Destructor
  virtual ~OptimisationProblem() {}

  /// Compute the objective function :math:`f(x)`
  virtual double f(const PETScVector& x) = 0;

  /// Compute the Hessian :math:`J(x)=f''(x)` and the gradient
  /// :math:`F(x)=f'(x)` together. Called before requesting
  /// F or J.
  /// NOTE: This function is deprecated. Use variant with
  /// preconditioner
  virtual void form(PETScMatrix& A, PETScVector& b, const PETScVector& x)
  {
    // NOTE: Deprecation mechanism
    _called = true;
  }

  /// Function called by the solver before requesting F, J or J_pc.
  /// This can be used to compute F, J and J_pc together. Preconditioner
  /// matrix P can be left empty so that A is used instead
  virtual void form(PETScMatrix& A, PETScMatrix& P, PETScVector& b,
                    const PETScVector& x)
  {
    // Do nothing if not supplied by the user

    // NOTE: Deprecation mechanism
    form(A, b, x);
    if (!_called)
    {
      // deprecated form(A, b, x) was not called which means that user
      // overloaded the deprecated method
      deprecation("NonlinearProblem::form(A, b, x)", "2017.1.0dev",
                  "Use NonlinearProblem::form(A, P, b, x)");
    }
    _called = false;
  }

  /// Compute the gradient :math:`F(x) = f'(x)`
  virtual void F(PETScVector& b, const PETScVector& x) = 0;

  /// Compute the Hessian :math:`J(x) = f''(x)`
  virtual void J(PETScMatrix& A, const PETScVector& x) = 0;

  /// Compute J_pc used to precondition J. Not implementing this
  /// or leaving P empty results in system matrix A being used
  /// to construct preconditioner.
  ///
  /// Note that if nonempty P is not assembled on first call
  /// then a solver implementation may throw away P and not
  /// call this routine ever again.
  virtual void J_pc(PETScMatrix& P, const PETScVector& x)
  {
    // Do nothing if not supplied by the user
  }

private:
  // NOTE: Deprecation mechanism
  bool _called;
};
}

#endif
