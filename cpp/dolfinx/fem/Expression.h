// Copyright (C) 2020 - 2021 Jack S. Hale and Michal Habera
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/array2d.h>
#include <dolfinx/fem/utils.h>
#include <functional>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace mesh
{
class Mesh;
}

namespace fem
{
template <typename T>
class Constant;

/// Represents a mathematical expression evaluated at a pre-defined set
/// of points on the reference cell. This class closely follows the
/// concept of a UFC Expression.
///
/// This functionality can be used to evaluate a gradient of a Function
/// at quadrature points in all cells. This evaluated gradient can then
/// be used as input in to a non-FEniCS function that calculates a
/// material constitutive model.

template <typename T>
class Expression
{
public:
  /// Create Expression
  ///
  /// @param[in] coefficients Coefficients in the Expression
  /// @param[in] constants Constants in the Expression
  /// @param[in] mesh
  /// @param[in] X points on reference cell, number of points rows and
  /// tdim cols
  /// @param[in] fn function for tabulating expression
  /// @param[in] value_size size of expression evaluated at single point
  Expression(
      const std::vector<std::shared_ptr<const fem::Function<T>>>& coefficients,
      const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants,
      const std::shared_ptr<const mesh::Mesh>& mesh, const array2d<double>& X,
      const std::function<void(T*, const T*, const T*, const double*,
                               const int*, const uint8_t*, uint32_t)>
          fn,
      const std::vector<int>& value_shape,
      const std::vector<int>& num_argument_dofs)
      : _coefficients(coefficients), _constants(constants), _mesh(mesh), _x(X),
        _fn(fn), _value_shape(value_shape),
        _num_argument_dofs(num_argument_dofs)
  {
    // Do nothing
  }

  /// Move constructor
  Expression(Expression&& form) = default;

  /// Destructor
  virtual ~Expression() = default;

  /// Access coefficients
  const std::vector<std::shared_ptr<const fem::Function<T>>>&
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

  /// Get function for tabulate_expression.
  /// @param[out] fn Function to tabulate expression.
  const std::function<void(T*, const T*, const T*, const double*, const int*,
                           const uint8_t*, uint32_t)>&
  get_tabulate_expression() const
  {
    return _fn;
  }

  /// Access constants
  /// @return Vector of attached constants with their names. Names are
  ///   used to set constants in user's c++ code. Index in the vector is
  ///   the position of the constant in the original (nonsimplified) form.
  const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants() const
  {
    return _constants;
  }

  /// Get mesh
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const { return _mesh; }

  /// Get evaluation points on reference cell
  /// @return Evaluation points
  const array2d<double>& x() const { return _x; }

  /// Get value size
  /// @return value_size
  const int value_size() const
  {
    return std::accumulate(_value_shape.begin(), _value_shape.end(), 1,
                           std::multiplies<int>());
  }

  /// Get value shape
  const std::vector<int>& value_shape() const { return _value_shape; }

  /// Get number of points
  /// @return number of points
  const std::size_t num_points() const { return _x.shape[0]; }

  /// Get number of degrees-of-freedom for arguments
  const std::vector<int>& num_argument_dofs() const
  {
    return _num_argument_dofs;
  }

  /// Scalar type (T).
  using scalar_type = T;

  /// Evaluate Expression at active cells
  array2d<T> eval(const tcb::span<const std::int32_t>& active_cells)
  {
    // Extract data from Expression
    assert(_mesh);

    // Prepare coefficients
    const array2d<T> coeffs = dolfinx::fem::pack_coefficients(*this);

    // Prepare constants
    const std::vector<T> constant_values = dolfinx::fem::pack_constants(*this);

    // Prepare cell geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = _mesh->geometry().dofmap();
    const fem::CoordinateElement& cmap = _mesh->geometry().cmap();

    // FIXME: Add proper interface for num coordinate dofs
    const int num_dofs_g = x_dofmap.num_links(0);
    const array2d<double>& x_g = _mesh->geometry().x();

    // Create data structures used in evaluation
    const int gdim = _mesh->geometry().dim();
    array2d<double> coordinate_dofs(num_dofs_g, gdim);

    // Iterate over cells and 'assemble' into values
    int num_all_argument_dofs
        = std::accumulate(num_argument_dofs().begin(),
                          num_argument_dofs().end(), 1, std::multiplies<int>());
    std::vector<T> values_local(num_points() * value_size() * num_all_argument_dofs,
                            0);

    // Dummy values, not required for cell expressions
    const int entity_local_index = 0;
    const uint8_t quadrature_permutation = 0;

    // Allocate memory for all results
    array2d<T> values(active_cells.size(), values_local.size());

    for (std::size_t i = 0; i < active_cells.size(); ++i)
    {
      const std::int32_t c = active_cells[i];
      assert(c < x_dofmap.num_nodes());
      auto x_dofs = x_dofmap.links(c);
      for (int j = 0; j < num_dofs_g; ++j)
      {
        const auto x_dof = x_dofs[j];
        for (int k = 0; k < gdim; ++k)
          coordinate_dofs(j, k) = x_g(x_dof, k);
      }

      auto coeff_cell = coeffs.row(c);
      std::fill(values_local.begin(), values_local.end(), 0.0);
      _fn(values_local.data(), coeff_cell.data(), constant_values.data(),
          coordinate_dofs.data(), &entity_local_index, &quadrature_permutation,
          0);

      for (std::size_t j = 0; j < values_local.size(); ++j)
        values(i, j) = values_local[j];
    }

    return values;
  }

private:
  // Function spaces (one for each argument)
  std::vector<std::shared_ptr<const fem::FunctionSpace>> _function_spaces;

  // Coefficients associated with the Expression
  std::vector<std::shared_ptr<const fem::Function<T>>> _coefficients;

  // Constants associated with the Expression
  std::vector<std::shared_ptr<const fem::Constant<T>>> _constants;

  // Function to evaluate the Expression
  std::function<void(T*, const T*, const T*, const double*, const int*,
                     const uint8_t*, uint32_t)>
      _fn;

  // Evaluation points on reference cell
  array2d<double> _x;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // Shape of the evaluated expression
  std::vector<int> _value_shape;

  // Number of degrees-of-freedom for arguments
  std::vector<int> _num_argument_dofs;

};
} // namespace fem
} // namespace dolfinx
