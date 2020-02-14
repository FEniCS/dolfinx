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
#include <petscsys.h>
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
  FiniteElement(const ufc_finite_element& ufc_element);

  /// Destructor
  virtual ~FiniteElement() = default;

  /// String identifying the finite element
  /// @return Element signature
  std::string signature() const;

  /// Cell shape
  /// @return Element cell shape
  mesh::CellType cell_shape() const;

  /// Dimension of the finite element function space
  /// @return Dimension of the finite element space
  int space_dimension() const;

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

  // FIXME: Is this well-defined? What does it do on non-simplex
  // elements?
  /// Return the maximum polynomial degree
  std::size_t degree() const;

  /// The finite element family
  /// @return The string of the finite element family
  std::string family() const;

  /// Evaluate all basis functions at given point in reference cell
  // reference_values[num_points][num_dofs][reference_value_size]
  void evaluate_reference_basis(
      Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
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
      const bool* edge_reflections, const bool* face_reflections,
      const std::uint8_t* face_rotations) const;

  /// Push basis function (derivatives) forward to physical element
  void transform_reference_basis_derivatives(
      Eigen::Tensor<double, 4, Eigen::RowMajor>& values, std::size_t order,
      const Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& X,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, 1>>& detJ,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
      const bool* edge_reflections, const bool* face_reflections,
      const std::uint8_t* face_rotations) const;

  /// Check if reference coordinates for dofs are defined
  /// @return True if the dof coordinates are available
  bool has_dof_reference_coordinates() const noexcept;

  /// Tabulate the reference coordinates of all dofs on an element
  /// @return The coordinates of all dofs on the reference cell
  const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>&
  dof_reference_coordinates() const;

  /// Map values of field from physical to reference space which has
  /// been evaluated at points given by dof_reference_coordinates()
  void transform_values(
      PetscScalar* reference_values,
      const Eigen::Ref<const Eigen::Array<PetscScalar, Eigen::Dynamic,
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

  int _tdim, _space_dim, _value_size, _reference_value_size, _degree;

  // Dof coordinates on the reference element
  bool _has_refX;
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
                    const double*, const double*, const double*, const bool*,
                    const bool*, const std::uint8_t*)>
      _transform_reference_basis_derivatives;

  std::function<int(ufc_scalar_t*, const ufc_scalar_t*, const double*,
                    const ufc_coordinate_mapping*)>
      _transform_values;
};
} // namespace dolfinx::fem
