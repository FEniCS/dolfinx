// Copyright (C) 2008-2013 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfin/common/types.h>
#include <memory>
#include <ufc.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace dolfin
{

namespace fem
{

/// This is a wrapper for a UFC finite element (ufc::finite_element).

class FiniteElement
{
public:
  /// Create finite element from UFC finite element (data may be shared)
  /// @param element (ufc::finite_element)
  ///  UFC finite element
  FiniteElement(std::shared_ptr<const ufc_finite_element> element);

  /// Destructor
  virtual ~FiniteElement() {}

  //--- Direct wrappers for ufc::finite_element ---

  /// Return a string identifying the finite element
  /// @return std::string
  std::string signature() const
  {
    assert(_ufc_element);
    assert(_ufc_element->signature);
    return _ufc_element->signature;
  }

  // FIXME: Avoid exposing UFC enum
  /// Return the cell shape
  /// @return ufc::shape
  ufc_shape cell_shape() const
  {
    assert(_ufc_element);
    return _ufc_element->cell_shape;
  }

  /// Return the topological dimension of the cell shape
  /// @return std::size_t
  std::size_t topological_dimension() const
  {
    assert(_ufc_element);
    return _ufc_element->topological_dimension;
  }

  /// Return the dimension of the finite element function space
  /// @return std::size_t
  std::size_t space_dimension() const
  {
    assert(_ufc_element);
    return _ufc_element->space_dimension;
  }

  /// Return the value size, e.g. 1 for a scalar function, 2 for a 2D
  /// vector
  std::size_t value_size() const
  {
    assert(_ufc_element);
    return _ufc_element->value_size;
  }

  /// Return the value size, e.g. 1 for a scalar function, 2 for a 2D
  /// vector
  std::size_t reference_value_size() const
  {
    assert(_ufc_element);
    return _ufc_element->reference_value_size;
  }

  /// Return the rank of the value space
  std::size_t value_rank() const
  {
    assert(_ufc_element);
    return _ufc_element->value_rank;
  }

  /// Return the dimension of the value space for axis i
  std::size_t value_dimension(std::size_t i) const
  {
    assert(_ufc_element);
    return _ufc_element->value_dimension(i);
  }

  // FIXME: Is this well-defined? What does it do on non-simplex
  // elements?
  /// Return the maximum polynomial degree
  std::size_t degree() const
  {
    assert(_ufc_element);
    return _ufc_element->degree;
  }

  /// Return the finite elemeent family
  std::string family() const
  {
    assert(_ufc_element);
    return _ufc_element->family;
  }

  /// Evaluate all basis functions at given point in reference cell
  // reference_values[num_points][num_dofs][reference_value_size]
  void evaluate_reference_basis(
      Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
      const Eigen::Ref<const EigenRowArrayXXd> X) const
  {
    assert(_ufc_element);
    std::size_t num_points = X.rows();
    _ufc_element->evaluate_reference_basis(reference_values.data(), num_points,
                                           X.data());
  }

  /// Push basis functions forward to physical element
  void transform_reference_basis(
      Eigen::Tensor<double, 3, Eigen::RowMajor>& values,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& reference_values,
      const Eigen::Ref<const EigenRowArrayXXd> X,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      const Eigen::Ref<const EigenArrayXd> detJ,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& K) const
  {
    assert(_ufc_element);
    std::size_t num_points = X.rows();
    _ufc_element->transform_reference_basis_derivatives(
        values.data(), 0, num_points, reference_values.data(), X.data(),
        J.data(), detJ.data(), K.data(), 1);
  }

  /// Push basis function (derivatives) forward to physical element
  void transform_reference_basis_derivatives(
      Eigen::Tensor<double, 4, Eigen::RowMajor>& values, std::size_t order,
      const Eigen::Tensor<double, 4, Eigen::RowMajor>& reference_values,
      const Eigen::Ref<const EigenRowArrayXXd> X,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      const Eigen::Ref<const EigenArrayXd> detJ,
      const Eigen::Tensor<double, 3, Eigen::RowMajor>& K) const
  {
    assert(_ufc_element);
    std::size_t num_points = X.rows();
    _ufc_element->transform_reference_basis_derivatives(
        values.data(), order, num_points, reference_values.data(), X.data(),
        J.data(), detJ.data(), K.data(), 1);
  }

  /// Tabulate the reference coordinates of all dofs on an element
  ///
  /// @return    reference_coordinates (EigenRowArrayXXd)
  ///         The coordinates of all dofs on the reference cell.
  const EigenRowArrayXXd& dof_reference_coordinates() const { return _refX; }

  // FIXME: pass coordinate map
  /// TODO: Remove? Document. See:
  /// ffc/uflacs/backends/ufc/evaluatedof.py:_change_variables()
  void map_dofs(double* values, const Eigen::Ref<const EigenRowArrayXXd>& vals,
                const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
  {
    assert(_ufc_element);
    _ufc_element->map_dofs(values, vals.data(), coordinate_dofs.data(), 1,
                           nullptr);
  }

  /// Return the number of sub elements (for a mixed element)
  /// @return std::size_t
  ///   number of sub-elements
  std::size_t num_sub_elements() const
  {
    assert(_ufc_element);
    return _ufc_element->num_sub_elements;
  }

  //--- DOLFIN-specific extensions of the interface ---

  /// Return simple hash of the signature string
  std::size_t hash() const { return _hash; }

  /// Create a new finite element for sub element i (for a mixed
  /// element)
  std::unique_ptr<FiniteElement> create_sub_element(std::size_t i) const;

  /// Create a new class instance
  std::unique_ptr<FiniteElement> create() const;

  /// Extract sub finite element for component
  std::shared_ptr<FiniteElement>
  extract_sub_element(const std::vector<std::size_t>& component) const;

private:
  // UFC finite element
  std::shared_ptr<const ufc_finite_element> _ufc_element;

  // Dof coordinates on the reference element
  EigenRowArrayXXd _refX;

  // Recursively extract sub finite element
  static std::shared_ptr<FiniteElement>
  extract_sub_element(const FiniteElement& finite_element,
                      const std::vector<std::size_t>& component);

  // Simple hash of the signature string
  std::size_t _hash;
};
}
}
