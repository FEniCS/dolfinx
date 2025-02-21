// Copyright (C) 2020-2025 Jack S. Hale, Michal Habera and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "Function.h"
#include <algorithm>
#include <array>
#include <concepts>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <memory>
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::fem
{
template <dolfinx::scalar T, std::floating_point U>
class Function;

/// @brief An Expression represents a mathematical expression evaluated
/// at a pre-defined points on a reference cell.
///
/// An Expression can be evaluated using ::tabulate_expression.
///
/// An example of Expression use is to evaluate the gradient of a
/// Function at quadrature points in cells. The evaluated gradient can
/// then be used as input in to a non-FEniCS function that evaluates a
/// material constitutive model.
///
/// @tparam T The scalar type.
/// @tparam U The mesh geometry scalar type.
template <dolfinx::scalar T,
          std::floating_point U = dolfinx::scalar_value_type_t<T>>
class Expression
{
public:
  /// @brief Scalar type.
  ///
  /// Field type for the Expression, e.g. `double`,
  /// `std::complex<float>`, etc.
  using scalar_type = T;

  /// Geometry type of the points.
  using geometry_type = U;

  /// @brief Create an Expression.
  ///
  /// @note Users should prefer the fem::create_expression factory
  /// functions for creating an Expression.
  ///
  /// @param[in] coefficients Coefficients in the Expression.
  /// @param[in] constants Constants in the Expression.
  /// @param[in] X Points on the reference cell, `shape=(number of
  /// points, tdim)` and storage is row-major.
  /// @param[in] Xshape Shape of `X`.
  /// @param[in] fn Function for tabulating the Expression.
  /// @param[in] value_shape Shape of Expression evaluated at single
  /// point.
  /// @param[in] argument_space Function space for Argument. Used to
  /// computed a 1-form expression, e.g. can be used to create a matrix
  /// that when applied to a degree-of-freedom vector gives the
  /// expression values at the evaluation points.
  Expression(
      const std::vector<std::shared_ptr<
          const Function<scalar_type, geometry_type>>>& coefficients,
      const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
          constants,
      std::span<const geometry_type> X, std::array<std::size_t, 2> Xshape,
      std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                         const geometry_type*, const int*, const uint8_t*, void*)>
          fn,
      const std::vector<std::size_t>& value_shape,
      std::shared_ptr<const FunctionSpace<geometry_type>> argument_space
      = nullptr)
      : _coefficients(coefficients), _constants(constants),
        _x_ref(std::vector<geometry_type>(X.begin(), X.end()), Xshape), _fn(fn),
        _value_shape(value_shape), _argument_space(argument_space)
  {
    for (auto& c : _coefficients)
    {
      assert(c);
      if (c->function_space()->mesh()
          != _coefficients.front()->function_space()->mesh())
      {
        throw std::runtime_error("Coefficients not all defined on same mesh.");
      }
    }
  }

  /// Move constructor
  Expression(Expression&& e) = default;

  /// Destructor
  virtual ~Expression() = default;

  /// @brief Argument function space.
  /// @return Argument function space, nullptr if there is no argument.
  std::shared_ptr<const FunctionSpace<geometry_type>> argument_space() const
  {
    return _argument_space;
  };

  /// @brief Expression coefficients.
  /// @return List of attached coefficients.
  const std::vector<
      std::shared_ptr<const Function<scalar_type, geometry_type>>>&
  coefficients() const
  {
    return _coefficients;
  }

  /// @brief Expression constants.
  ///
  /// @return List of attached constants.
  const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
  constants() const
  {
    return _constants;
  }

  /// @brief Offset for each coefficient expansion array on a cell.
  ///
  /// Used to pack data for multiple coefficients into a flat array. The
  /// last entry is the size required to store all coefficients.
  /// @return The offsets.
  std::vector<int> coefficient_offsets() const
  {
    std::vector<int> n{0};
    for (auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// @brief Function for tabulating the Expression.
  const std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                           const geometry_type*, const int*, const uint8_t*, void*)>&
  kernel() const
  {
    return _fn;
  }

  /// @brief Value size of the Expression result.
  int value_size() const
  {
    return std::reduce(_value_shape.begin(), _value_shape.end(), 1,
                       std::multiplies{});
  }

  /// @brief Value shape of of Expression result (at a point),
  const std::vector<std::size_t>& value_shape() const { return _value_shape; }

  /// @brief Evaluation point coordinates on the reference cell.
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>> X() const
  {
    return _x_ref;
  }

private:
  // Function space for Argument
  std::shared_ptr<const FunctionSpace<geometry_type>> _argument_space;

  // Coefficients associated with the Expression
  std::vector<std::shared_ptr<const Function<scalar_type, geometry_type>>>
      _coefficients;

  // Constants associated with the Expression
  std::vector<std::shared_ptr<const Constant<scalar_type>>> _constants;

  // Function to evaluate the Expression
  std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                     const geometry_type*, const int*, const uint8_t*, void*)>
      _fn;

  // Shape of the evaluated Expression
  std::vector<std::size_t> _value_shape;

  // Evaluation points on reference cell
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>> _x_ref;
};
} // namespace dolfinx::fem
