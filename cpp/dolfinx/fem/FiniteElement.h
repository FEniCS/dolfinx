// Copyright (C) 2008-2020 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/common/types.h>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <unsupported/Eigen/CXX11/Tensor>
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
  void evaluate_reference_basis(
      Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
      const Eigen::Ref<const Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& X) const;

  /// Evaluate all basis function derivatives of given order at given points in
  /// reference cell
  // reference_value_derivatives[num_points][num_dofs][reference_value_size][num_derivatives]
  void evaluate_reference_basis_derivatives(
      Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values, int order,
      const Eigen::Ref<const Eigen::Array<
          double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>& X) const;

  /// Push basis functions forward to physical element
  void transform_reference_basis(
      Eigen::Tensor<double, 3, Eigen::RowMajor>& values,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& X,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>& detJ,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& K) const;

  /// Push basis function (derivatives) forward to physical element
  void transform_reference_basis_derivatives(
      Eigen::Tensor<double, 4, Eigen::RowMajor>& values, std::size_t order,
      const Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& X,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>& detJ,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& K) const;

  /// @todo Add documentation
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
  dof_coordinates(int cell_perm) const;

  /// Map values of field from physical to reference space which has
  /// been evaluated at points given by dof_reference_coordinates()
  void transform_values(
      ufc_scalar_t* reference_values,
      const Eigen::Ref<const Eigen::Array<ufc_scalar_t, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          physical_values,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          coordinate_dofs) const;

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
  ///  @return True is interpolation is an identity operation
  bool interpolation_ident() const noexcept;

  /// Points on the reference cell at which an expression need to be
  /// evaluated in order to interpolate the expression in the finite
  /// element space. For Lagrange elements the points will just be the
  /// nodal positions. For other elements the points will typically be
  /// the quadrature points used to evaluate moment degrees of freedom.
  /// @return Points on the reference cell. Shape is (num_points, tdim).
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  interpolation_points() const noexcept;

  /// @todo Document shape/layout of @p values
  /// @todo Make the interpolating dofs in/out argument for efficiency
  /// as this function is often called from within tight loops
  /// @todo Consider handling block size > 1
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
  void interpolate(const Eigen::Array<ufc_scalar_t, Eigen::Dynamic,
                                      Eigen::Dynamic, Eigen::RowMajor>& values,
                   const std::uint32_t cell_permutation,
                   Eigen::Array<ufc_scalar_t, Eigen::Dynamic, 1>& dofs) const;

  /// @todo Expand on when permutation data might be required
  ///
  /// Check if cell permutation data is required for this element
  /// @return True if cell permutation data is required
  bool needs_permutation_data() const noexcept;

  /// Apply permutation to some data
  ///
  /// @param[in] data The data to be transformed
  /// @param[in] cell_permutation Permutation data fro the cell
  /// @param[in] block_size The block_size of the input data
  void apply_dof_transformation(double* data,
                                const std::uint32_t cell_permutation,
                                const int block_size) const;

  /// Apply reverse permutation to some data
  ///
  /// @param[in] data The data to be transformed
  /// @param[in] cell_permutation Permutation data fro the cell
  /// @param[in] block_size The block_size of the input data
  void apply_reverse_dof_transformation(double* data,
                                        const std::uint32_t cell_permutation,
                                        const int block_size) const;

private:
  std::string _signature, _family;

  mesh::CellType _cell_shape;

  int _tdim, _space_dim, _value_size, _reference_value_size;

  // Dof coordinates on the reference element
  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> _refX;

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

  std::function<int(double*, int, int, const double*, const double*,
                    const double*, const double*, const double*)>
      _transform_reference_basis_derivatives;

  std::function<int(ufc_scalar_t*, const ufc_scalar_t*, const double*,
                    const ufc_coordinate_mapping*)>
      _transform_values;

  std::function<int(double*, const std::uint32_t, const int)>
      _apply_dof_transformation;

  std::function<int(double*, const std::uint32_t, const int)>
      _apply_reverse_dof_transformation;

  // Block size for VectorElements and TensorElements. This gives the
  // number of DOFs colocated at each point.
  int _bs;

  // True if interpolation is indetity, i.e. call to
  // _interpolate_into_cell is not required
  bool _interpolation_is_ident;

  bool _needs_permutation_data;

  int _basix_element_handle;
};
} // namespace dolfinx::fem
