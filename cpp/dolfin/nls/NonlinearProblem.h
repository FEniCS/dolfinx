// Copyright (C) 2005-2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfin
{

namespace la
{
class PETScMatrix;
class PETScVector;
} // namespace la

namespace nls
{

/// This is a base class for nonlinear problems which can return the
/// nonlinear function F(u) and its Jacobian J = dF(u)/du.

class NonlinearProblem
{
public:
  /// Constructor
  NonlinearProblem() = default;

  /// Destructor
  virtual ~NonlinearProblem() {}

  /// Function called by Newton solver before requesting F, J or J_pc.
  /// This can be used to compute F, J and J_pc together.
  /// Note: the vector x is not const as this function is commonly used
  /// to update ghost entries before assembly.
  virtual void form(la::PETScVector& x)
  {
    // Do nothing if not supplied by the user
  }

  /// Compute F at current point x
  virtual la::PETScVector* F(const la::PETScVector& x) = 0;

  /// Compute J = F' at current point x
  virtual la::PETScMatrix* J(const la::PETScVector& x) = 0;

  /// Compute J_pc used to precondition J. Not implementing this
  /// or leaving P empty results in system matrix A being used
  /// to construct preconditioner.
  virtual la::PETScMatrix* P(const la::PETScVector& x) { return nullptr; }
};
} // namespace nls
} // namespace dolfin