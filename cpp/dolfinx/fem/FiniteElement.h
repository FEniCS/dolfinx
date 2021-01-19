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
  std::string signature() const;

  /// Cell shape
  /// @return Element cell shape
  mesh::CellType cell_shape() const;

  /// Dimension of the finite element function space
  /// @return Dimension of the finite element space
  int space_dimension() const;

  /// Block size of the finite element function space. For VectorElements and
  /// TensorElements, this is the number of DOFs colocated at each DOF point.
  /// For other elements, this is always 1.
  /// @return Block size of the finite element space
  int block_size() const;

  /// The value size, e.g. 1 for a scalar function, 2 for a 2D vector
  /// @return The value size
  int value_size() const;

  /// The value size, e.g. 1 for a scalar function, 2 for a 2D vector
  /// for the reference element
  /// @return The value size for the reference element
  int reference_value_size() const;

  /// Rank of the value space
  /// @return The value rank
  int value_rank() const;

  /// Return the dimension of the value space for axis i
  int value_dimension(int i) const;

  /// The finite element family
  /// @return The string of the finite element family
  std::string family() const;

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
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
      const std::uint32_t permutation_info) const;

  /// Push basis function (derivatives) forward to physical element
  void transform_reference_basis_derivatives(
      Eigen::Tensor<double, 4, Eigen::RowMajor>& values, std::size_t order,
      const Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& X,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>& detJ,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
      const std::uint32_t permutation_info) const;

  /// Tabulate the reference coordinates of all dofs on an element
  /// @return The coordinates of all dofs on the reference cell
  ///
  /// @note Throws an exception if dofs cannot be associated with points
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  dof_reference_coordinates() const;

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

  /// Return the number of sub elements (for a mixed element)
  int num_sub_elements() const;

  /// Return simple hash of the signature string
  std::size_t hash() const;

  /// Extract sub finite element for component
  std::shared_ptr<const FiniteElement>
  extract_sub_element(const std::vector<int>& component) const;

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

  // Functions for basis and derivatives evaluation
  std::function<int(double*, int, const double*)> _evaluate_reference_basis;

  std::function<int(double*, int, int, const double*)>
      _evaluate_reference_basis_derivatives;

  std::function<int(double*, int, int, const double*, const double*,
                    const double*, const double*, const double*,
                    const std::uint32_t)>
      _transform_reference_basis_derivatives;

  std::function<int(ufc_scalar_t*, const ufc_scalar_t*, const double*,
                    const ufc_coordinate_mapping*)>
      _transform_values;

  // Block size for VectorElements and TensorElements
  // This gives the number of DOFs colocated at each point
  int _block_size;
};
} // namespace dolfinx::fem
