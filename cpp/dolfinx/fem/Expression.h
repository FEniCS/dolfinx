// Copyright (C) 2020 - 2021 Jack S. Hale and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/mesh/Mesh.h>
#include "Function.h"
#include <functional>
#include <utility>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::fem
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
  /// Create an Expression
  ///
  /// @param[in] function_space Function space for Argument
  /// @param[in] coefficients Coefficients in the Expression
  /// @param[in] constants Constants in the Expression
  /// @param[in] mesh
  /// @param[in] X points on reference cell, number of points rows and
  /// tdim cols
  /// @param[in] fn function for tabulating expression
  /// @param[in] value_shape shape of expression evaluated at single point
  Expression(
      const std::shared_ptr<const fem::FunctionSpace> function_space,
      const std::vector<std::shared_ptr<const fem::Function<T>>>& coefficients,
      const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants,
      const std::shared_ptr<const mesh::Mesh>& mesh,
      const xt::xtensor<double, 2>& X,
      const std::function<void(T*, const T*, const T*, const double*,
                               const int*, const uint8_t*)>
          fn,
      const std::vector<int>& value_shape)
      : _function_space(function_space), _coefficients(coefficients),
        _constants(constants), _mesh(mesh), _X(X), _fn(fn),
        _value_shape(value_shape)
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

  /// Evaluate the expression on cells
  /// @param[in] active_cells Cells on which to evaluate the Expression
  /// @param[out] values A 2D array to store the result. Caller
  /// is responsible for correct sizing which should be (num_cells,
  /// num_points * value_size * num_all_argument_dofs columns).
  template <typename U>
  void eval(const xtl::span<const std::int32_t>& active_cells, U& values) const
  {
    // Extract data from Expression
    assert(_mesh);

    // Prepare coefficients and constants
    const auto [coeffs, cstride] = pack_coefficients(*this, active_cells);
    const std::vector<T> constant_data = pack_constants(*this);

    // Prepare cell geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = _mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    const xt::xtensor<double, 2>& x_g = _mesh->geometry().x();

    // Create data structures used in evaluation
    std::vector<double> coordinate_dofs(3 * num_dofs_g);

    const int num_argument_dofs
        = _function_space->dofmap()->element_dof_layout()->num_dofs();
    std::vector<T> values_local(num_points() * value_size() * num_argument_dofs,
                                0);

    const bool needs_transformation_data
        = element->needs_dof_transformations();
    xtl::span<const std::uint32_t> cell_info;
    if (needs_transformation_data)
    {
      mesh->topology_mutable().create_entity_permutations();
      cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    }

    // Iterate over cells and 'assemble' into values
    for (std::size_t c = 0; c < active_cells.size(); ++c)
    {
      const std::int32_t cell = active_cells[c];

      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(xt::row(x_g, x_dofs[i]).cbegin(), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }

      const T* coeff_cell = coeffs.data() + cell * cstride;
      std::fill(values_local.begin(), values_local.end(), 0.0);
      _fn(values_local.data(), coeff_cell, constant_data.data(),
         coordinate_dofs.data(), nullptr, nullptr);

      for (std::size_t j = 0; j < values_local.size(); ++j)
        values(c, j) = values_local[j];
    }
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
  const xt::xtensor<double, 2>& X() const { return _X; }

  /// Get value size
  /// @return value_size
  int value_size() const
  {
    return std::accumulate(_value_shape.begin(), _value_shape.end(), 1,
                           std::multiplies<int>());
  }

  /// Get value shape
  const std::vector<int>& value_shape() const { return _value_shape; }

  /// Get number of evaluation points in cell
  /// @return number of points in cell
  std::size_t num_points() const { return _X.shape(0); }

  /// Scalar type (T)
  using scalar_type = T;

private:
  // Function space for Argument
  std::shared_ptr<const fem::FunctionSpace> _function_space;

  // Coefficients associated with the Expression
  std::vector<std::shared_ptr<const fem::Function<T>>> _coefficients;

  // Constants associated with the Expression
  std::vector<std::shared_ptr<const fem::Constant<T>>> _constants;

  // Function to evaluate the Expression
  std::function<void(T*, const T*, const T*, const double*, const int*,
                     const uint8_t*)>
      _fn;

  // Evaluation points on reference cell
  xt::xtensor<double, 2> _X;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // Shape of the evaluated expression
  std::vector<int> _value_shape;

};
} // namespace dolfinx::fem
