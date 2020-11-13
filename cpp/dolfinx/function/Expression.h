// Copyright (C) 2020 Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/function/evaluate.h>
#include <functional>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace mesh
{
class Mesh;
}

namespace function
{
template <typename T>
class Constant;

/// Represents a mathematical expression evaluated at a pre-defined set of
/// points on the reference cell. This class closely follows the concept of a
/// UFC Expression.
///
/// This functionality can be used to evaluate a gradient of a Function at
/// quadrature points in all cells. This evaluated gradient can then be used as
/// input in to a non-FEniCS function that calculates a material constitutive
/// model.

template <typename T>
class Expression
{
public:
  /// Create Expression.
  ///
  /// @param[in] coefficients Coefficients in the Expression
  /// @param[in] constants Constants in the Expression
  /// @param[in] mesh
  /// @param[in] x points on reference cell, number of points rows
  ///   and tdim cols
  /// @param[in] fn function for tabulating expression
  /// @param[in] value_size size of expression evaluated at single point
  Expression(
      const std::vector<std::shared_ptr<const function::Function<T>>>&
          coefficients,
      const std::vector<std::shared_ptr<const function::Constant<T>>>&
          constants,
      const std::shared_ptr<const mesh::Mesh>& mesh,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& x,
      const std::function<void(T*, const T*, const T*, const double*)> fn,
      const std::size_t value_size)
      : _coefficients(coefficients), _constants(constants), _mesh(mesh), _x(x),
        _fn(fn), _value_size(value_size)
  {
    // Do nothing
  }

  /// Move constructor
  Expression(Expression&& form) = default;

  /// Destructor
  virtual ~Expression() = default;

  /// Access coefficients
  const std::vector<std::shared_ptr<const function::Function<T>>>&
  coefficients() const
  {
    return _coefficients;
  }

  /// Offset for each coefficient expansion array on a cell. Used to
  /// pack data for multiple coefficients in a flat array. The last
  /// entry is the size required to store all coefficients.
  std::vector<int> coefficient_offsets() const
  {
    std::vector<int> n{0};
    for (const auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// Evaluate the expression on cells
  /// @param[in] active_cells Cells on which to evaluate the Expression
  /// @param[in,out] values To store the result. Caller responsible for correct
  ///   sizing which should be num_cells rows by num_points*value_size columns.
  void
  eval(const std::vector<std::int32_t>& active_cells,
       Eigen::Ref<
           Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
           values) const
  {
    function::eval(values, *this, active_cells);
  }

  /// Get function for tabulate_expression.
  /// @param[out] fn Function to tabulate expression.
  const std::function<void(T*, const T*, const T*, const double*)>&
  get_tabulate_expression() const
  {
    return _fn;
  }

  /// Access constants
  /// @return Vector of attached constants with their names. Names are
  ///   used to set constants in user's c++ code. Index in the vector is
  ///   the position of the constant in the original (nonsimplified) form.
  const std::vector<std::shared_ptr<const function::Constant<T>>>&
  constants() const
  {
    return _constants;
  }

  /// Get mesh
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const { return _mesh; }

  /// Get evaluation points on reference cell
  /// @return Evaluation points
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  x() const
  {
    return _x;
  }

  /// Get value size
  /// @return value_size
  const std::size_t value_size() const { return _value_size; }

  /// Get number of points
  /// @return number of points
  const Eigen::Index num_points() const { return _x.rows(); }

  /// Scalar type (T).
  using scalar_type = T;

private:
  // Coefficients associated with the Expression
  std::vector<std::shared_ptr<const function::Function<T>>> _coefficients;

  // Constants associated with the Expression
  std::vector<std::shared_ptr<const function::Constant<T>>> _constants;

  // Function to evaluate the Expression
  std::function<void(T*, const T*, const T*, const double*)> _fn;

  // Evaluation points on reference cell
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _x;

  // The mesh.
  std::shared_ptr<const mesh::Mesh> _mesh;

  // Evaluation size
  std::size_t _value_size;
};
} // namespace function
} // namespace dolfinx
