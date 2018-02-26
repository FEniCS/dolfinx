// Copyright (C) 2005-2008 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfin
{

// Forward declarations
class PETScMatrix;
class PETScVector;

namespace nls
{

/// This is a base class for nonlinear problems which can return the
/// nonlinear function F(u) and its Jacobian J = dF(u)/du.

class NonlinearProblem
{
public:
  /// Constructor
  NonlinearProblem() {}

  /// Destructor
  virtual ~NonlinearProblem() {}

  /// Function called by Newton solver before requesting F, J or J_pc.
  /// This can be used to compute F, J and J_pc together. Preconditioner
  /// matrix P can be left empty so that A is used instead
  virtual void form(PETScMatrix& A, PETScMatrix& P, PETScVector& b,
                    const PETScVector& x)
  {
    // Do nothing if not supplied by the user
  }

  /// Compute F at current point x
  virtual void F(PETScVector& b, const PETScVector& x) = 0;

  /// Compute J = F' at current point x
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
};
}
}