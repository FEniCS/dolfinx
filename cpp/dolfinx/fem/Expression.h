// Copyright (C) 2020 Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Function.h"
#include <dolfinx/common/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <utility>
#include <vector>
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
  /// @param[in] coefficients Coefficients in the Expression
  /// @param[in] constants Constants in the Expression
  /// @param[in] mesh
  /// @param[in] X points on reference cell, shape=(num_points, tdim)
  /// @param[in] fn function for tabulating expression
  /// @param[in] value_size size of expression evaluated at single point
  Expression(
      const std::vector<std::shared_ptr<const Function<T>>>& coefficients,
      const std::vector<std::shared_ptr<const Constant<T>>>& constants,
      const std::shared_ptr<const mesh::Mesh>& mesh,
      const xt::xtensor<double, 2>& X,
      const std::function<void(T*, const T*, const T*, const double*)> fn,
      std::size_t value_size)
      : _coefficients(coefficients), _constants(constants), _mesh(mesh), _x(X),
        _fn(fn), _value_size(value_size)
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
    for (auto& c : _coefficients)
    {
      if (!c)
        throw std::runtime_error("Not all form coefficients have been set.");
      n.push_back(n.back() + c->function_space()->element()->space_dimension());
    }
    return n;
  }

  /// Evaluate the expression on cells
  /// @param[in] cells Cells on which to evaluate the Expression
  /// @param[out] values A 2D array to store the result. Caller
  /// responsible for correct sizing which should be (num_cells,
  /// num_points, value_size), flattened (row-major) rp (num_cells,
  /// num_points * value_size).
  template <typename U>
  void eval(const xtl::span<const std::int32_t>& cells, U& values) const
  {
    static_assert(std::is_same<T, typename U::value_type>::value,
                  "Expression and array types must be the same");

    // Extract data from Expression
    assert(_mesh);

    // Prepare coefficients and constants
    const auto [coeffs, cstride] = pack_coefficients(*this, cells);
    const std::vector<T> constant_data = pack_constants(*this);
    const auto& fn = this->get_tabulate_expression();

    // Prepare cell geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = _mesh->geometry().dofmap();

    // FIXME: Add proper interface for num coordinate dofs
    const std::size_t num_dofs_g = x_dofmap.num_links(0);
    xtl::span<const double> x_g = _mesh->geometry().x();

    // Create data structures used in evaluation
    std::vector<double> coordinate_dofs(3 * num_dofs_g);

    // Iterate over cells and 'assemble' into values
    std::vector<T> values_e(_x.shape(0) * this->value_size(), 0);
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
      const std::int32_t cell = cells[c];

      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        common::impl::copy_N<3>(std::next(x_g.begin(), 3 * x_dofs[i]),
                                std::next(coordinate_dofs.begin(), 3 * i));
      }

      const T* coeff_cell = coeffs.data() + c * cstride;
      std::fill(values_e.begin(), values_e.end(), 0.0);
      fn(values_e.data(), coeff_cell, constant_data.data(),
         coordinate_dofs.data());

      for (std::size_t j = 0; j < values_e.size(); ++j)
        values(c, j) = values_e[j];
    }
  }

  /// Get function for tabulate_expression.
  /// @return Function to tabulate expression.
  const std::function<void(T*, const T*, const T*, const double*)>&
  get_tabulate_expression() const
  {
    return _fn;
  }

  /// Access constants
  /// @return Vector of attached constants with their names. Names are
  /// used to set constants in user's c++ code. Index in the vector is
  /// the position of the constant in the original (nonsimplified) form.
  const std::vector<std::shared_ptr<const fem::Constant<T>>>& constants() const
  {
    return _constants;
  }

  /// Get mesh
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const { return _mesh; }

  /// Evaluation points on the reference cell
  /// @return Evaluation points (shape=(num_points, tdim))
  const xt::xtensor<double, 2>& x() const { return _x; }

  /// Get value size
  /// @return value_size
  std::size_t value_size() const { return _value_size; }

  /// Scalar type (T)
  using scalar_type = T;

private:
  // Coefficients associated with the Expression
  std::vector<std::shared_ptr<const fem::Function<T>>> _coefficients;

  // Constants associated with the Expression
  std::vector<std::shared_ptr<const fem::Constant<T>>> _constants;

  // Function to evaluate the Expression
  std::function<void(T*, const T*, const T*, const double*)> _fn;

  // Evaluation points on reference cell
  xt::xtensor<double, 2> _x;

  // The mesh.
  std::shared_ptr<const mesh::Mesh> _mesh;

  // Evaluation size
  std::size_t _value_size;
};
} // namespace dolfinx::fem
