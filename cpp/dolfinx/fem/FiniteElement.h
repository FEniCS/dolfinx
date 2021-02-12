// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/span.hpp>
#include <dolfinx/common/types.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <numeric>
#include <vector>

struct ufc_coordinate_mapping;
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
  FiniteElement(const FiniteElement& element) = default;

  /// Move constructor
  FiniteElement(FiniteElement&& element) = default;

  /// Destructor
  virtual ~FiniteElement() = default;

  /// Copy assignment
  FiniteElement& operator=(const FiniteElement& element) = default;

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
  void evaluate_reference_basis(std::vector<double>& values,
                                const common::array2d<double>& X) const;

  /// Evaluate all basis function derivatives of given order at given points in
  /// reference cell
  // reference_value_derivatives[num_points][num_dofs][reference_value_size][num_derivatives]
  void
  evaluate_reference_basis_derivatives(std::vector<double>& reference_values,
                                       int order,
                                       const common::array2d<double>& X) const;

  /// Push basis functions forward to physical element
  void transform_reference_basis(std::vector<double>& values,
                                 const std::vector<double>& reference_values,
                                 const common::array2d<double>& X,
                                 const std::vector<double>& J,
                                 const tcb::span<const double>& detJ,
                                 const std::vector<double>& K) const;

  /// Push basis function (derivatives) forward to physical element
  void transform_reference_basis_derivatives(
      std::vector<double>& values, std::size_t order,
      const std::vector<double>& reference_values,
      const common::array2d<double>& X, const std::vector<double>& J,
      const tcb::span<const double>& detJ, const std::vector<double>& K) const;

  /// Get the number of sub elements (for a mixed element)
  /// @return the Number of sub elements
  int num_sub_elements() const noexcept;

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
  common::array2d<double> interpolation_points() const;

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
  /// @param[in] cell_permutation Permutation data for the cell
  /// @param[out] dofs The element degrees of freedom (interpolants) of
  /// the expression. The call must allocate the space. Is has
  template <typename T>
  constexpr void interpolate(const common::array2d<T>& values,
                             std::uint32_t cell_permutation,
                             tcb::span<T> dofs) const
  {
    const std::size_t rows = _space_dim / _bs;
    assert(_space_dim % _bs == 0);
    assert(dofs.size() == rows);

    // Compute dofs = Pi * x (matrix-vector multiply)
    assert(_interpolation_matrix.size() % rows == 0);
    const std::size_t cols = _interpolation_matrix.size() / rows;
    for (std::size_t i = 0; i < rows; ++i)
    {
      // Dot product between row i of the matrix and 'values'
      dofs[i] = std::transform_reduce(
          std::next(_interpolation_matrix.begin(), i * cols),
          std::next(_interpolation_matrix.begin(), i * cols + cols),
          values.data(), T(0.0));
    }

    _apply_dof_transformation_to_scalar(dofs.data(), cell_permutation, 1);
  }

  /// @todo Expand on when permutation data might be required
  ///
  /// Check if cell permutation data is required for this element
  /// @return True if cell permutation data is required
  bool needs_permutation_data() const noexcept;

  /// Apply permutation to some data
  ///
  /// @param[in,out] data The data to be transformed
  /// @param[in] cell_permutation Permutation data fro the cell
  /// @param[in] block_size The block_size of the input data
  template <typename T>
  void apply_dof_transformation(T* data, std::uint32_t cell_permutation,
                                int block_size) const
  {
    if constexpr (std::is_same<T, double>::value)
      _apply_dof_transformation(data, cell_permutation, block_size);
    else
      _apply_dof_transformation_to_scalar(data, cell_permutation, block_size);
  }

private:
  std::string _signature, _family;

  mesh::CellType _cell_shape;

  int _tdim, _space_dim, _value_size, _reference_value_size;

  // List of sub-elements (if any)
  std::vector<std::shared_ptr<const FiniteElement>> _sub_elements;

  // Recursively extract sub finite element
  static std::shared_ptr<const FiniteElement>
  extract_sub_element(const FiniteElement& finite_element,
                      const std::vector<int>& component);

  // Simple hash of the signature string
  std::size_t _hash;

  // Dimension of each value space
  std::vector<int> _value_dimension;

  std::function<int(double*, const std::uint32_t, const int)>
      _apply_dof_transformation;

  std::function<int(ufc_scalar_t*, const std::uint32_t, const int)>
      _apply_dof_transformation_to_scalar;

  // Block size for VectorElements and TensorElements. This gives the
  // number of DOFs colocated at each point.
  int _bs;

  // True if interpolation is indentity, i.e. call to
  // _interpolate_into_cell is not required
  bool _interpolation_is_ident;

  // True if element needs dof permutation
  bool _needs_permutation_data;

  // The basix element identifier
  int _basix_element_handle;

  // The interpolation matrix. It has shape (dim,
  // num_interpolation_points*basix_value_size). Note that _space_dim =
  // _bs * dim. Storage is row-major
  std::vector<double> _interpolation_matrix;
};
} // namespace dolfinx::fem
