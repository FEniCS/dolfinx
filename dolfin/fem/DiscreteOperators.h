// Copyright (C) 2015 Garth N. Wells
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

#ifndef __DOLFIN_DISCRETE_OPERATORS_H
#define __DOLFIN_DISCRETE_OPERATORS_H

#include <memory>

namespace dolfin
{

  class FunctionSpace;
  class GenericMatrix;

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

    /// Build the discrete gradient operator A that takes a w \in H^1
    /// (P1, nodal Lagrange) to v \in H(curl) (lowest order Nedelec),
    /// i.e. v = Aw. V0 is the H(curl) space, and V1 is the P1
    /// Lagrange space.
    ///
    /// @param[in] FunctionSpace V0
    ///  H(curl) space
    /// @param[in] FunctionSpace V1
    ///  P1 Lagrange space
    ///
    /// @return GenericMatrix
    static std::shared_ptr<GenericMatrix>
      build_gradient(const FunctionSpace& V0, const FunctionSpace& V1);

  };
}

#endif
