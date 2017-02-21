// Copyright (C) 2005-2008 Garth N. Wells
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
// Modified by Anders Logg, 2008.
//
// First added:  2005-10-24
// Last changed: 2011-01-14

#ifndef __NONLINEAR_PROBLEM_H
#define __NONLINEAR_PROBLEM_H

namespace dolfin
{

  // Forward declarations
  class GenericMatrix;
  class GenericVector;

  /// This is a base class for nonlinear problems which can return the
  /// nonlinear function F(u) and its Jacobian J = dF(u)/du.

  class NonlinearProblem
  {
  public:

    /// Constructor
    NonlinearProblem() : _called(false) {}

    /// Destructor
    virtual ~NonlinearProblem() {}

    /// Function called by Newton solver before requesting F or J.
    /// This can be used to compute F and J together.
    /// NOTE: This function is deprecated. Use variant with
    /// preconditioner
    virtual void form(GenericMatrix& A, GenericVector& b,
                      const GenericVector& x)
    {
      // NOTE: Deprecation mechanism
      _called = true;
    }

    /// Function called by Newton solver before requesting F, J or J_pc.
    /// This can be used to compute F, J and J_pc together. Preconditioner
    /// matrix P can be left empty so that A is used instead
    virtual void form(GenericMatrix& A, GenericMatrix& P, GenericVector& b,
                      const GenericVector& x)
    {
      // Do nothing if not supplied by the user

      // NOTE: Deprecation mechanism
      form(A, b, x);
      if (!_called)
      {
        // deprecated form(A, b, x) was not called which means that user
        // overloaded the deprecated method
        deprecation("NonlinearProblem::form(A, b, x)",
                    "2017.1.0dev",
                    "Use NonlinearProblem::form(A, P, b, x)");
      }
      _called = false;
    }

    /// Compute F at current point x
    virtual void F(GenericVector& b, const GenericVector& x) = 0;

    /// Compute J = F' at current point x
    virtual void J(GenericMatrix& A, const GenericVector& x) = 0;

    /// Compute J_pc used to precondition J. Not implementing this
    /// or leaving P empty results in system matrix A being used
    /// to construct preconditioner
    virtual void J_pc(GenericMatrix& P, const GenericVector& x)
    {
      // Do nothing if not supplied by the user
    }

  private:

    // NOTE: Deprecation mechanism
    bool _called;

  };

}

#endif
