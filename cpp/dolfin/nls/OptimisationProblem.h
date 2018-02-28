// Copyright (C) 2014 Tianyi Li
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "NonlinearProblem.h"

namespace dolfin
{
namespace la
{
class PETScMatrix;
class PETScVector;
}

namespace nls
{

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
  virtual double f(const la::PETScVector& x) = 0;

  /// Function called by the solver before requesting F, J or J_pc.
  /// This can be used to compute F, J and J_pc together. Preconditioner
  /// matrix P can be left empty so that A is used instead
  virtual void form(la::PETScMatrix& A, la::PETScMatrix& P, la::PETScVector& b,
                    const la::PETScVector& x)
  {
    // Do nothing if not supplied by the user
  }

  /// Compute the gradient :math:`F(x) = f'(x)`
  virtual void F(la::PETScVector& b, const la::PETScVector& x) = 0;

  /// Compute the Hessian :math:`J(x) = f''(x)`
  virtual void J(la::PETScMatrix& A, const la::PETScVector& x) = 0;

  /// Compute J_pc used to precondition J. Not implementing this
  /// or leaving P empty results in system matrix A being used
  /// to construct preconditioner.
  ///
  /// Note that if nonempty P is not assembled on first call
  /// then a solver implementation may throw away P and not
  /// call this routine ever again.
  virtual void J_pc(la::PETScMatrix& P, const la::PETScVector& x)
  {
    // Do nothing if not supplied by the user
  }
};
}
}