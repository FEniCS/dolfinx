// Copyright (C) 2020 - 2021 Jack S. Hale and Michal Habera
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Function.h"
#include <algorithm>
#include <array>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <span>
#include <utility>
#include <vector>

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
  template <typename X, typename = void>
  struct scalar_value_type
  {
    typedef X value_type;
  };
  template <typename X>
  struct scalar_value_type<X, std::void_t<typename X::value_type>>
  {
    typedef typename X::value_type value_type;
  };
  using scalar_value_type_t = typename scalar_value_type<T>::value_type;

public:
  /// @brief Create an Expression
  ///
  /// @note Users should prefer the create_expression factory functions
  ///
  /// @param[in] coefficients Coefficients in the Expression
  /// @param[in] constants Constants in the Expression
  /// @param[in] mesh
  /// @param[in] X points on reference cell, shape=(number of points,
  /// tdim) and storage is row-major.
  /// @param[in] Xshape Shape of `X`.
  /// @param[in] fn function for tabulating expression
  /// @param[in] value_shape shape of expression evaluated at single point
  /// @param[in] argument_function_space Function space for Argument
  Expression(
      const std::vector<std::shared_ptr<const Function<T>>>& coefficients,
      const std::vector<std::shared_ptr<const Constant<T>>>& constants,
      std::span<const double> X, std::array<std::size_t, 2> Xshape,
      const std::function<void(T*, const T*, const T*,
                               const scalar_value_type_t*, const int*,
                               const uint8_t*)>
          fn,
      const std::vector<int>& value_shape,
      std::shared_ptr<const mesh::Mesh> mesh = nullptr,
      std::shared_ptr<const FunctionSpace> argument_function_space = nullptr)
      : _coefficients(coefficients), _constants(constants), _mesh(mesh),
        _x_ref(std::vector<double>(X.begin(), X.end()), Xshape), _fn(fn),
        _value_shape(value_shape),
        _argument_function_space(argument_function_space)
  {
    // Extract mesh from argument's function space
    if (!_mesh and argument_function_space)
      _mesh = argument_function_space->mesh();
    if (argument_function_space and _mesh != argument_function_space->mesh())
      throw std::runtime_error("Incompatible mesh");
    if (!_mesh)
    {
      throw std::runtime_error(
          "No mesh could be associated with the Expression.");
    }
  }

  /// Move constructor
  Expression(Expression&& form) = default;

  /// Destructor
  virtual ~Expression() = default;

  /// Get argument function space
  /// @return The argument function space, nullptr if there is no argument.
  std::shared_ptr<const FunctionSpace> argument_function_space() const
  {
    return _argument_function_space;
  };

  /// Get coefficients
  /// @return Vector of attached coefficients
  const std::vector<std::shared_ptr<const Function<T>>>& coefficients() const
  {
    return _coefficients;
  }

  /// Get constants
  /// @return Vector of attached constants with their names. Names are
  ///   used to set constants in user's c++ code. Index in the vector is
  ///   the position of the constant in the original (nonsimplified) form.
  const std::vector<std::shared_ptr<const Constant<T>>>& constants() const
  {
    return _constants;
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

  /// @brief Evaluate the expression on cells
  /// @param[in] cells Cells on which to evaluate the Expression
  /// @param[out] values A 2D array to store the result. Caller
  /// is responsible for correct sizing which should be (num_cells,
  /// num_points * value_size * num_all_argument_dofs columns).
  /// @param[in] vshape The shape of `values` (row-major storage).
  void eval(std::span<const std::int32_t> cells, std::span<T> values,
            std::array<std::size_t, 2> vshape) const
  {
    // Extract data from Expression
    assert(_mesh);

    // Prepare coefficients and constants
    const auto [coeffs, cstride] = pack_coefficients(*this, cells);
    const std::vector<T> constant_data = pack_constants(*this);
    const auto& fn = this->get_tabulate_expression();

    // Prepare cell geometry
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = _mesh->geometry().dofmap();
    const std::size_t num_dofs_g = _mesh->geometry().cmap().dim();
    std::span<const double> x_g = _mesh->geometry().x();

    // Create data structures used in evaluation
    std::vector<scalar_value_type_t> coordinate_dofs(3 * num_dofs_g);

    int num_argument_dofs = 1;
    std::span<const std::uint32_t> cell_info;
    std::function<void(const std::span<T>&,
                       const std::span<const std::uint32_t>&, std::int32_t,
                       int)>
        dof_transform_to_transpose
        = [](const std::span<T>&, const std::span<const std::uint32_t>&,
             std::int32_t, int)
    {
      // Do nothing
    };

    if (_argument_function_space)
    {
      num_argument_dofs
          = _argument_function_space->dofmap()->element_dof_layout().num_dofs();
      auto element = _argument_function_space->element();

      assert(element);
      if (element->needs_dof_transformations())
      {
        _mesh->topology_mutable().create_entity_permutations();
        cell_info = std::span(_mesh->topology().get_cell_permutation_info());
        dof_transform_to_transpose
            = element
                  ->template get_dof_transformation_to_transpose_function<T>();
      }
    }

    const int size0 = _x_ref.second[0] * value_size();
    std::vector<T> values_local(size0 * num_argument_dofs, 0);
    const std::span<T> _values_local(values_local);

    // Iterate over cells and 'assemble' into values
    for (std::size_t c = 0; c < cells.size(); ++c)
    {
      const std::int32_t cell = cells[c];

      auto x_dofs = x_dofmap.links(cell);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                    std::next(coordinate_dofs.begin(), 3 * i));
      }

      const T* coeff_cell = coeffs.data() + c * cstride;
      std::fill(values_local.begin(), values_local.end(), 0);
      _fn(values_local.data(), coeff_cell, constant_data.data(),
          coordinate_dofs.data(), nullptr, nullptr);

      dof_transform_to_transpose(_values_local, cell_info, c, size0);
      for (std::size_t j = 0; j < values_local.size(); ++j)
        values[c * vshape[1] + j] = values_local[j];
    }
  }

  /// Get function for tabulate_expression.
  /// @return fn Function to tabulate expression.
  const std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                           const int*, const uint8_t*)>&
  get_tabulate_expression() const
  {
    return _fn;
  }

  /// Get mesh
  /// @return The mesh
  std::shared_ptr<const mesh::Mesh> mesh() const { return _mesh; }

  /// Get value size
  /// @return value_size
  int value_size() const
  {
    return std::reduce(_value_shape.begin(), _value_shape.end(), 1,
                       std::multiplies{});
  }

  /// Get value shape
  /// @return value shape
  const std::vector<int>& value_shape() const { return _value_shape; }

  /// @brief Evaluation points on the reference cell
  /// @return Evaluation points
  std::pair<std::vector<double>, std::array<std::size_t, 2>> X() const
  {
    return _x_ref;
  }

  /// Scalar type (T)
  using scalar_type = T;

private:
  // Function space for Argument
  std::shared_ptr<const FunctionSpace> _argument_function_space;

  // Coefficients associated with the Expression
  std::vector<std::shared_ptr<const Function<T>>> _coefficients;

  // Constants associated with the Expression
  std::vector<std::shared_ptr<const Constant<T>>> _constants;

  // Function to evaluate the Expression
  std::function<void(T*, const T*, const T*, const scalar_value_type_t*,
                     const int*, const uint8_t*)>
      _fn;

  // The mesh
  std::shared_ptr<const mesh::Mesh> _mesh;

  // Shape of the evaluated expression
  std::vector<int> _value_shape;

  // Evaluation points on reference cell. Synonymous with X in public interface.
  std::pair<std::vector<double>, std::array<std::size_t, 2>> _x_ref;
};
} // namespace dolfinx::fem
