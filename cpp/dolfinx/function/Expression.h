// Copyright (C) 2020 Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Constant.h"
#include "evaluate.h"
#include <dolfinx/fem/FormCoefficients.h>
#include <functional>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace fem
{
template <typename T>
class FormCoefficients;
}

namespace mesh
{
class Mesh;
}

namespace function
{
template <typename T>
class Constant;

/// Represents a mathematical expression evaluated at a pre-defined set of
/// points on the reference cell. This class closely follows the concept of a UFC
/// Expression.

template <typename T>
class Expression
{
public:
  /// Create Expression.
  Expression(
      const fem::FormCoefficients<T>& coefficients,
      const std::vector<
          std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>
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

  /// Create Expression. coefficients, constants and fn must be set later by
  /// caller.
  Expression(
      const std::shared_ptr<mesh::Mesh>& mesh,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& x,
      const std::size_t value_size)
      : Expression(fem::FormCoefficients<T>({}), {}, mesh, x, nullptr,
                   value_size)
  {
    // Do nothing
  }

  /// Move constructor
  Expression(Expression&& form) = default;

  /// Destructor
  virtual ~Expression() = default;

  /// Access coefficients
  fem::FormCoefficients<T>& coefficients() { return _coefficients; }

  /// Access coefficients (const version)
  const fem::FormCoefficients<T>& coefficients() const { return _coefficients; }

  /// Evaluate the expression on cells
  /// @param[in] active_cells Cells on which to evaluate the Expression
  /// @param[in,out] values To store the result. Caller responsible for correct
  /// sizing.
  void
  eval(const std::vector<std::int32_t>& active_cells,
       Eigen::Ref<
           Eigen::Array<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
           values) const
  {
    function::eval(values, *this, active_cells);
  }

  /// Register the function for tabulate_expression.
  /// @param[in] fn Function to tabulate expression.
  void set_tabulate_expression(
      const std::function<void(T*, const T*, const T*, const double*)> fn)
  {
    _fn = fn;
  }

  /// Register the function for tabulate_expression.
  /// @param[in] fn Function to tabulate expression.
  const std::function<void(T*, const T*, const T*, const double*)>&
  get_tabulate_expression() const
  {
    return _fn;
  }

  /// Set coefficient with given number (shared pointer version)
  /// @param[in] coefficients Map from coefficient index to the
  ///   coefficient
  void set_coefficients(
      const std::map<int, std::shared_ptr<const function::Function<T>>>&
          coefficients)
  {
    for (const auto& c : coefficients)
      _coefficients.set(c.first, c.second);
  }

  /// Set coefficient with given name (shared pointer version)
  /// @param[in] coefficients Map from coefficient name to the
  ///   coefficient
  void set_coefficients(
      const std::map<std::string, std::shared_ptr<const function::Function<T>>>&
          coefficients)
  {
    for (const auto& c : coefficients)
      _coefficients.set(c.first, c.second);
  }

  /// Set constants based on their names
  ///
  /// This method is used in command-line workflow, when users set
  /// constants to the form in cpp file.
  ///
  /// Names of the constants must agree with their names in UFL file.
  void set_constants(
      const std::map<std::string, std::shared_ptr<const function::Constant<T>>>&
          constants)
  {
    for (auto const& constant : constants)
    {
      // Find matching string in existing constants
      const std::string name = constant.first;
      const auto it = std::find_if(
          _constants.begin(), _constants.end(),
          [&](const std::pair<
              std::string, std::shared_ptr<const function::Constant<T>>>& q) {
            return (q.first == name);
          });
      if (it != _constants.end())
        it->second = constant.second;
      else
        throw std::runtime_error("Constant '" + name + "' not found in form");
    }
  }

  /// Set constants based on their order (without names)
  ///
  /// This method is used in Python workflow, when constants are
  /// automatically attached to the expression based on their order in the
  /// original expression.
  ///
  /// The order of constants must match their order in original ufl
  /// expression.
  void
  set_constants(const std::vector<std::shared_ptr<const function::Constant<T>>>&
                    constants)
  {
    // TODO: Why this check? Should resize as necessary.
    // if (constants.size() != _constants.size())
    // throw std::runtime_error("Incorrect number of constants.");
    _constants.resize(constants.size());

    // Loop over each constant that user wants to attach
    for (std::size_t i = 0; i < constants.size(); ++i)
    {
      // In this case, the constants don't have names
      _constants[i] = std::pair("", constants[i]);
    }
  }

  /// Access constants
  /// @return Vector of attached constants with their names. Names are
  ///   used to set constants in user's c++ code. Index in the vector is
  ///   the position of the constant in the original (nonsimplified) form.
  const std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>&
  constants() const
  {
    return _constants;
  }

  /// Check if all constants associated with the expression have been set
  /// @return True if all Form constants have been set
  bool all_constants_set() const
  {
    for (const auto& constant : _constants)
      if (!constant.second)
        return false;
    return true;
  }

  /// Return names of any constants that have not been set
  /// @return Names of unset constants
  std::set<std::string> get_unset_constants() const
  {
    std::set<std::string> unset;
    for (const auto& constant : _constants)
      if (!constant.second)
        unset.insert(constant.first);
    return unset;
  }

  /// Get mesh
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const { return _mesh; }

  /// Get evaluation points
  /// @return Evaluation points
  const Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>& x() const
  {
    return _x;
  }

  /// Get value size
  /// @return value_size
  const std::size_t value_size() const { return _value_size; }

  /// Get number of points
  /// @return number of points
  const Eigen::Index num_points() const { return _x.rows(); }

private:
  // Coefficients associated with the Expression
  fem::FormCoefficients<T> _coefficients;

  // Constants associated with the Expression
  std::vector<
      std::pair<std::string, std::shared_ptr<const function::Constant<T>>>>
      _constants;

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
