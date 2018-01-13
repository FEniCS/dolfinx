// Copyright (C) 2010 Marie E. Rognes
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/function/Expression.h>
#include <vector>

namespace ufc
{
class cell;
}

namespace dolfin
{
class Function;

/// A _SpecialFacetFunction_ is a representation of a global
/// function that is in P(f) for each _Facet_ f in a _Mesh_ for some
/// _FunctionSpace_ P

class SpecialFacetFunction : public Expression
{
public:
  /// Create (scalar-valued) SpecialFacetFunction
  ///
  /// *Arguments*
  ///     f_e (std::vector<_Function_>)
  ///        Separate _Function_s for each facet
  explicit SpecialFacetFunction(std::vector<Function>& f_e);

  /// Create (vector-valued) SpecialFacetFunction
  ///
  /// *Arguments*
  ///     f_e (std::vector<_Function_>)
  ///        Separate _Function_s for each facet
  ///
  ///     dim (int)
  ///         The value-dimension of the Functions
  SpecialFacetFunction(std::vector<Function>& f_e, std::size_t dim);

  /// Create (tensor-valued) SpecialFacetFunction
  ///
  /// *Arguments*
  ///     f_e (std::vector<_Function_>)
  ///        Separate _Function_s for each facet
  ///
  ///     value_shape (std::vector<std::size_t>)
  ///         The values-shape of the Functions
  SpecialFacetFunction(std::vector<Function>& f_e,
                       std::vector<std::size_t> value_shape);

  /// Evaluate SpecialFacetFunction (cf _Expression_.eval)
  /// Evaluate function for given cell
  void eval(Eigen::Ref<Eigen::VectorXd> values,
            Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& cell) const;

  /// Extract sub-function i
  ///
  /// *Arguments*
  ///     i (int)
  ///        component
  ///
  /// *Returns*
  ///     _Function_
  Function& operator[](std::size_t i) const;

private:
  std::vector<Function>& _f_e;
};
}

