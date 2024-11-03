// Copyright (C) 2020-2021 Jack S. Hale and Michal Habera.
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
#include <span>
#include <utility>
#include <vector>

namespace dolfinx::fem
{
template <dolfinx::scalar T>
class Constant;

/// @brief Represents a mathematical expression evaluated at a
/// pre-defined set of points on the reference cell.
///
//// This class closely follows the concept of a UFC Expression.
///
/// The functionality can be used to evaluate a gradient of a Function
/// at quadrature points in all cells. This evaluated gradient can then
/// be used as input in to a non-FEniCS function that calculates a
/// material constitutive model.
///
/// @tparam T The scalar type
/// @tparam U The mesh geometry scalar type
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
  /// @note Users should prefer the @ref create_expression factory
  /// functions.
  ///
  /// @param[in] coefficients Coefficients in the Expression.
  /// @param[in] constants Constants in the Expression
  /// @param[in] X Points on the reference cell, `shape=(number of
  /// points, tdim)` and storage is row-major.
  /// @param[in] Xshape Shape of `X`.
  /// @param[in] fn Function for tabulating the Expression.
  /// @param[in] value_shape Shape of Expression evaluated at single
  /// point.
  /// @param[in] argument_function_space Function space for Argument.
  Expression(
      const std::vector<std::shared_ptr<
          const Function<scalar_type, geometry_type>>>& coefficients,
      const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
          constants,
      std::span<const geometry_type> X, std::array<std::size_t, 2> Xshape,
      std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                         const geometry_type*, const int*, const uint8_t*)>
          fn,
      const std::vector<std::size_t>& value_shape,
      std::shared_ptr<const FunctionSpace<geometry_type>>
          argument_function_space
      = nullptr)
      : _coefficients(coefficients), _constants(constants),
        _x_ref(std::vector<geometry_type>(X.begin(), X.end()), Xshape), _fn(fn),
        _value_shape(value_shape),
        _argument_function_space(argument_function_space)
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

  /// @brief Get argument function space.
  /// @return The argument function space, nullptr if there is no
  /// argument.
  std::shared_ptr<const FunctionSpace<geometry_type>>
  argument_function_space() const
  {
    return _argument_function_space;
  };

  /// @brief Get coefficients.
  /// @return Vector of attached coefficients.
  const std::vector<
      std::shared_ptr<const Function<scalar_type, geometry_type>>>&
  coefficients() const
  {
    return _coefficients;
  }

  /// @brief Get constants.
  /// @return Vector of attached constants with their names. Names are
  /// used to set constants in user's c++ code. Index in the vector is
  /// the position of the constant in the original (nonsimplified) form.
  const std::vector<std::shared_ptr<const Constant<scalar_type>>>&
  constants() const
  {
    return _constants;
  }

  /// @brief Offset for each coefficient expansion array on a cell.
  ///
  /// Used to pack data for multiple coefficients in a flat array. The
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

  /// @brief Evaluate Expression on cells or facets.
  ///
  /// @param[in] mesh Cells on which to evaluate the Expression.
  /// @param[in] entities List of entities to evaluate the expression
  /// on. This could be either a list of cells or a list of (cell, local
  /// @param[out] values A 2D array to store the result. Caller is
  /// responsible for correct sizing which should be `(num_cells,
  /// num_points * value_size * num_all_argument_dofs columns)`.
  /// facet index) tuples. Array is flattened per entity.
  /// @param[in] vshape The shape of `values` (row-major storage).
  void eval(const mesh::Mesh<geometry_type>& mesh,
            std::span<const std::int32_t> entities,
            std::span<scalar_type> values,
            std::array<std::size_t, 2> vshape) const
  {
    std::size_t estride;
    if (mesh.topology()->dim() == _x_ref.second[1])
      estride = 1;
    else if (mesh.topology()->dim() == _x_ref.second[1] + 1)
      estride = 2;
    else
      throw std::runtime_error("Invalid dimension of evaluation points.");

    // Prepare coefficients and constants
    auto [coeffs, cstride] = pack_coefficients(*this, entities, estride);
    std::vector<scalar_type> constant_data = pack_constants(*this);
    auto fn = this->get_tabulate_expression();

    // Prepare cell geometry
    auto x_dofmap = mesh.geometry().dofmap();

    // Get geometry data
    auto& cmap = mesh.geometry().cmap();

    std::size_t num_dofs_g = cmap.dim();
    auto x_g = mesh.geometry().x();

    // Create data structures used in evaluation
    std::vector<geometry_type> coord_dofs(3 * num_dofs_g);

    int num_argument_dofs = 1;
    std::span<const std::uint32_t> cell_info;
    std::function<void(std::span<scalar_type>, std::span<const std::uint32_t>,
                       std::int32_t, int)>
        post_dof_transform
        = [](std::span<scalar_type>, std::span<const std::uint32_t>,
             std::int32_t, int)
    {
      // Do nothing
    };

    if (_argument_function_space)
    {
      num_argument_dofs
          = _argument_function_space->dofmap()->element_dof_layout().num_dofs();
      auto element = _argument_function_space->element();
      num_argument_dofs *= _argument_function_space->dofmap()->bs();
      assert(element);
      if (element->needs_dof_transformations())
      {
        mesh.topology_mutable()->create_entity_permutations();
        cell_info = std::span(mesh.topology()->get_cell_permutation_info());
        post_dof_transform
            = element->template dof_transformation_right_fn<scalar_type>(
                doftransform::transpose);
      }
    }

    // Create get entity index function
    std::function<const std::int32_t*(std::span<const std::int32_t>,
                                      std::size_t)>
        get_entity_index
        = []([[maybe_unused]] std::span<const std::int32_t> entities,
             [[maybe_unused]] std::size_t idx) { return nullptr; };
    if (estride == 2)
    {
      get_entity_index
          = [](std::span<const std::int32_t> entities, std::size_t idx)
      { return entities.data() + 2 * idx + 1; };
    }

    // Iterate over cells and 'assemble' into values
    int size0 = _x_ref.second[0] * value_size();
    std::vector<scalar_type> values_local(size0 * num_argument_dofs, 0);
    for (std::size_t e = 0; e < entities.size() / estride; ++e)
    {
      std::int32_t entity = entities[e * estride];
      auto x_dofs = MDSPAN_IMPL_STANDARD_NAMESPACE::submdspan(
          x_dofmap, entity, MDSPAN_IMPL_STANDARD_NAMESPACE::full_extent);
      for (std::size_t i = 0; i < x_dofs.size(); ++i)
      {
        std::copy_n(std::next(x_g.begin(), 3 * x_dofs[i]), 3,
                    std::next(coord_dofs.begin(), 3 * i));
      }

      const scalar_type* coeff_cell = coeffs.data() + e * cstride;
      const int* entity_index = get_entity_index(entities, e);

      std::ranges::fill(values_local, 0);
      _fn(values_local.data(), coeff_cell, constant_data.data(),
          coord_dofs.data(), entity_index, nullptr);
      post_dof_transform(values_local, cell_info, e, size0);
      for (std::size_t j = 0; j < values_local.size(); ++j)
        values[e * vshape[1] + j] = values_local[j];
    }
  }

  /// @brief Get function for tabulate_expression.
  /// @return fn Function to tabulate expression.
  const std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                           const geometry_type*, const int*, const uint8_t*)>&
  get_tabulate_expression() const
  {
    return _fn;
  }

  /// @brief Get value size
  /// @return The value size.
  int value_size() const
  {
    return std::reduce(_value_shape.begin(), _value_shape.end(), 1,
                       std::multiplies{});
  }

  /// @brief Get value shape.
  /// @return The value shape.
  const std::vector<std::size_t>& value_shape() const { return _value_shape; }

  /// @brief Evaluation points on the reference cell.
  /// @return Evaluation points.
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>> X() const
  {
    return _x_ref;
  }

private:
  // Function space for Argument
  std::shared_ptr<const FunctionSpace<geometry_type>> _argument_function_space;

  // Coefficients associated with the Expression
  std::vector<std::shared_ptr<const Function<scalar_type, geometry_type>>>
      _coefficients;

  // Constants associated with the Expression
  std::vector<std::shared_ptr<const Constant<scalar_type>>> _constants;

  // Function to evaluate the Expression
  std::function<void(scalar_type*, const scalar_type*, const scalar_type*,
                     const geometry_type*, const int*, const uint8_t*)>
      _fn;

  // Shape of the evaluated expression
  std::vector<std::size_t> _value_shape;

  // Evaluation points on reference cell. Synonymous with X in public
  // interface.
  std::pair<std::vector<geometry_type>, std::array<std::size_t, 2>> _x_ref;
};
} // namespace dolfinx::fem
