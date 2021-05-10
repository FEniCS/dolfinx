// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <basix/finite-element.h>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>
#include <xtl/xspan.hpp>

struct ufc_finite_element;

namespace dolfinx::fem
{
/// Finite Element, containing the dof layout on a reference element, and
/// various methods for evaluating and transforming the basis.
class FiniteElement
{
public:
  /// Create finite element from UFC finite element
  /// @param[in] ufc_element UFC finite element
  explicit FiniteElement(const ufc_finite_element& ufc_element);

  /// Copy constructor
  FiniteElement(const FiniteElement& element) = delete;

  /// Move constructor
  FiniteElement(FiniteElement&& element) = default;

  /// Destructor
  virtual ~FiniteElement() = default;

  /// Copy assignment
  FiniteElement& operator=(const FiniteElement& element) = delete;

  /// Move assignment
  FiniteElement& operator=(FiniteElement&& element) = default;

  /// String identifying the finite element
  /// @return Element signature
  std::string signature() const noexcept;

  /// Cell shape
  /// @return Element cell shape
  mesh::CellType cell_shape() const noexcept;

  /// Dimension of the finite element function space
  /// @return Dimension of the finite element space
  int space_dimension() const noexcept;

  /// Block size of the finite element function space. For VectorElements and
  /// TensorElements, this is the number of DOFs colocated at each DOF point.
  /// For other elements, this is always 1.
  /// @return Block size of the finite element space
  int block_size() const noexcept;

  /// The value size, e.g. 1 for a scalar function, 2 for a 2D vector
  /// @return The value size
  int value_size() const noexcept;

  /// The value size, e.g. 1 for a scalar function, 2 for a 2D vector
  /// for the reference element
  /// @return The value size for the reference element
  int reference_value_size() const noexcept;

  /// Rank of the value space
  /// @return The value rank
  int value_rank() const noexcept;

  /// Return the dimension of the value space for axis i
  int value_dimension(int i) const;

  /// The finite element family
  /// @return The string of the finite element family
  std::string family() const noexcept;

  /// Evaluate all basis functions at given points in reference cell
  // reference_values[num_points][num_dofs][reference_value_size]
  void evaluate_reference_basis(xt::xtensor<double, 3>& values,
                                const xt::xtensor<double, 2>& X) const;

  /// Evaluate all basis function derivatives of given order at given points in
  /// reference cell
  // reference_value_derivatives[num_points][num_dofs][reference_value_size][num_derivatives]
  // void
  // evaluate_reference_basis_derivatives(std::vector<double>& reference_values,
  //                                      int order,
  //                                      const xt::xtensor<double, 2>& X)
  //                                      const;

  /// Push basis functions forward to physical element
  void transform_reference_basis(xt::xtensor<double, 3>& values,
                                 const xt::xtensor<double, 3>& reference_values,
                                 const xt::xtensor<double, 3>& J,
                                 const xtl::span<const double>& detJ,
                                 const xt::xtensor<double, 3>& K) const;

  /// Push basis function (derivatives) forward to physical element
  void transform_reference_basis_derivatives(
      std::vector<double>& values, std::size_t order,
      const std::vector<double>& reference_values, const array2d<double>& X,
      const std::vector<double>& J, const xtl::span<const double>& detJ,
      const std::vector<double>& K) const;

  /// Get the number of sub elements (for a mixed element)
  /// @return the Number of sub elements
  int num_sub_elements() const noexcept;

  /// Subelements (if any)
  const std::vector<std::shared_ptr<const FiniteElement>>&
  sub_elements() const noexcept;

  /// Return simple hash of the signature string
  std::size_t hash() const noexcept;

  /// Extract sub finite element for component
  std::shared_ptr<const FiniteElement>
  extract_sub_element(const std::vector<int>& component) const;

  /// Check if interpolation into the finite element space is an
  /// identity operation given the evaluation on an expression at
  /// specific points, i.e. the degree-of-freedom are equal to point
  /// evaluations. The function will return `true` for Lagrange
  /// elements.
  ///  @return True if interpolation is an identity operation
  bool interpolation_ident() const noexcept;

  /// Points on the reference cell at which an expression need to be
  /// evaluated in order to interpolate the expression in the finite
  /// element space. For Lagrange elements the points will just be the
  /// nodal positions. For other elements the points will typically be
  /// the quadrature points used to evaluate moment degrees of freedom.
  /// @return Points on the reference cell. Shape is (num_points, tdim).
  const xt::xtensor<double, 2>& interpolation_points() const;

  /// @todo Document shape/layout of @p values
  /// @todo Make the interpolating dofs in/out argument for efficiency
  /// as this function is often called from within tight loops
  /// @todo Consider handling block size > 1
  /// @todo Re-work for fields that require a pull-back, e.g. Piols
  /// mapped elements
  ///
  /// Interpolate a function in the finite element space on a cell.
  /// Given the evaluation of the function to be interpolated at points
  /// provided by @p FiniteElement::interpolation_points, it evaluates
  /// the degrees of freedom for the interpolant.
  ///
  /// @param[in] values The values of the function. It has shape
  /// (value_size, num_points), where `num_points` is the number of
  /// points given by FiniteElement::interpolation_points.
  /// @param[out] dofs The element degrees of freedom (interpolants) of
  /// the expression. The call must allocate the space. Is has
  template <typename T>
  constexpr void interpolate(const xt::xtensor<T, 2>& values,
                             xtl::span<T> dofs) const
  {
    if (!_element)
    {
      throw std::runtime_error("No underlying element for interpolation. "
                               "Cannot interpolate mixed elements directly.");
    }

    const std::size_t rows = _space_dim / _bs;
    assert(_space_dim % _bs == 0);
    assert(dofs.size() == rows);

    // Compute dofs = Pi * x (matrix-vector multiply)
    const xt::xtensor<double, 2>& Pi = _element->interpolation_matrix();
    assert(Pi.size() % rows == 0);
    const std::size_t cols = Pi.size() / rows;
    for (std::size_t i = 0; i < rows; ++i)
    {
      // Dot product between row i of the matrix and 'values'
      dofs[i] = std::transform_reduce(std::next(Pi.data(), i * cols),
                                      std::next(Pi.data(), i * cols + cols),
                                      values.data(), T(0.0));
    }
  }

  /// @todo Expand on when permutation data might be required
  ///
  /// Check if cell permutation data is required for this element
  /// @return True if cell permutation data is required
  bool needs_permutation_data() const noexcept;

  /// Apply permutation to some data
  ///
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_dof_transformation(xtl::span<T> data,
                                std::uint32_t cell_permutation,
                                int block_size) const
  {
    assert(_element);
    _element->apply_dof_transformation(data, block_size, cell_permutation);
  }

  /// Apply inverse transpose permutation to some data
  ///
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_inverse_transpose_dof_transformation(
      xtl::span<T> data, std::uint32_t cell_permutation, int block_size) const
  {
    assert(_element);
    _element->apply_inverse_transpose_dof_transformation(data, block_size,
                                                         cell_permutation);
  }

  /// Pull physical data back to the reference element.
  /// This passes the inputs directly into Basix's map_pull_back function.
  template <typename T>
  void
  map_pull_back(const xt::xtensor<T, 3>& u, const xt::xtensor<double, 3>& J,
                const xtl::span<const double>& detJ,
                const xt::xtensor<double, 3>& K, xt::xtensor<T, 3>& U) const
  {
    assert(_element);
    _element->map_pull_back_m(u, J, detJ, K, U);
  }

private:
  std::string _signature, _family;

  mesh::CellType _cell_shape;

  int _tdim, _space_dim, _value_size, _reference_value_size;

  // List of sub-elements (if any)
  std::vector<std::shared_ptr<const FiniteElement>> _sub_elements;

  // Simple hash of the signature string
  std::size_t _hash;

  // Dimension of each value space
  std::vector<int> _value_dimension;

  // Block size for VectorElements and TensorElements. This gives the
  // number of DOFs colocated at each point.
  int _bs;

  // True if element needs dof permutation
  bool _needs_permutation_data;

  // Basix Element (nullptr for mixed elements)
  std::unique_ptr<basix::FiniteElement> _element;
};
} // namespace dolfinx::fem
