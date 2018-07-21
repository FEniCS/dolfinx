// Copyright (C) 2015 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>

namespace dolfin
{

namespace function
{
class FunctionSpace;
}

namespace la
{
class PETScMatrix;
}

namespace fem
{

/// Discrete gradient operators providing derivatives of functions

/// This class computes discrete gradient operators (matrices) that
/// map derivatives of finite element functions into other finite
/// element spaces. An example of where discrete gradient operators
/// are required is the creation of algebraic multigrid solvers for
/// H(curl) and H(div) problems.

/// NOTE: This class is highly experimental and likely to change. It
/// will eventually be expanded to provide the discrete curl and
/// divergence.

class DiscreteOperators
{
public:
  /// Build the discrete gradient operator A that takes a \f$w \in H^1\f$
  /// (P1, nodal Lagrange) to \f$v \in H(curl)\f$ (lowest order Nedelec),
  /// i.e. v = Aw. V0 is the H(curl) space, and V1 is the P1
  /// Lagrange space.
  ///
  /// @param[in] V0 (function::FunctionSpace&)
  ///  H(curl) space
  /// @param[in] V1 (function::FunctionSpace&)
  ///  P1 Lagrange space
  ///
  /// @return la::PETScMatrix
  static std::shared_ptr<la::PETScMatrix>
  build_gradient(const function::FunctionSpace& V0,
                 const function::FunctionSpace& V1);
};
} // namespace fem
} // namespace dolfin
