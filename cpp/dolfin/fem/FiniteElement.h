// Copyright (C) 2008-2013 Anders Logg and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <boost/multi_array.hpp>
#include <dolfin/common/types.h>
#include <dolfin/log/log.h>
#include <memory>
#include <ufc.h>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;
}

namespace fem
{

/// This is a wrapper for a UFC finite element (ufc::finite_element).

class FiniteElement
{
public:
  /// Create finite element from UFC finite element (data may be shared)
  /// @param element (ufc::finite_element)
  ///  UFC finite element
  FiniteElement(std::shared_ptr<const ufc::finite_element> element);

  /// Destructor
  virtual ~FiniteElement() {}

  //--- Direct wrappers for ufc::finite_element ---

  /// Return a string identifying the finite element
  /// @return std::string
  std::string signature() const
  {
    dolfin_assert(_ufc_element);
    return _ufc_element->signature();
  }

  /// Return the cell shape
  /// @return ufc::shape
  ufc::shape cell_shape() const
  {
    dolfin_assert(_ufc_element);
    return _ufc_element->cell_shape();
  }

  /// Return the topological dimension of the cell shape
  /// @return std::size_t
  std::size_t topological_dimension() const
  {
    dolfin_assert(_ufc_element);
    return _ufc_element->topological_dimension();
  }

  /// Return the geometric dimension of the cell shape
  /// @return std::uint32_t
  virtual std::uint32_t geometric_dimension() const
  {
    dolfin_assert(_ufc_element);
    return _ufc_element->geometric_dimension();
  }

  /// Return the dimension of the finite element function space
  /// @return std::size_t
  std::size_t space_dimension() const
  {
    dolfin_assert(_ufc_element);
    return _ufc_element->space_dimension();
  }

  /// Return the rank of the value space
  std::size_t value_rank() const
  {
    dolfin_assert(_ufc_element);
    return _ufc_element->value_rank();
  }

  /// Return the dimension of the value space for axis i
  std::size_t value_dimension(std::size_t i) const
  {
    dolfin_assert(_ufc_element);
    return _ufc_element->value_dimension(i);
  }

  // /// Evaluate basis function i at given point in cell
  // void evaluate_basis(std::size_t i, double* values, const double* x,
  //                     const double* coordinate_dofs, int cell_orientation)
  //                     const
  // {
  //   dolfin_assert(_ufc_element);
  //   _ufc_element->evaluate_basis(i, values, x, coordinate_dofs,
  //                                cell_orientation);
  // }

  /// Evaluate all basis functions at given point in cell
  // void evaluate_reference_basis(double * reference_values,
  //                               std::size_t num_points,
  //                               const double * X) const final override
  // reference_values[num_points][num_dofs][reference_value_size]
  void
  evaluate_reference_basis(boost::multi_array<double, 3>& reference_values,
                           const Eigen::Ref<const EigenRowArrayXXd> X) const
  {
    assert(_ufc_element);
    std::size_t num_points = X.rows();
    _ufc_element->evaluate_reference_basis(reference_values.data(), num_points,
                                           X.data());
  }

  /// Push basis functions forward to physical element
  void transform_reference_basis(
      boost::multi_array<double, 3>& values,
      const boost::multi_array<double, 3>& reference_values,
      const Eigen::Ref<const EigenRowArrayXXd> X,
      const boost::multi_array<double, 3>& J,
      const Eigen::Ref<const EigenArrayXd> detJ,
      const boost::multi_array<double, 3>& K) const
  {
    assert(_ufc_element);
    std::size_t num_points = X.rows();
    _ufc_element->transform_reference_basis_derivatives(
        values.data(), 0, num_points, reference_values.data(), X.data(),
        J.data(), detJ.data(), K.data(), 1);
  }

  /// Push basis function (derivatives) forward to physical element
  void transform_reference_basis_derivatives(
      boost::multi_array<double, 4>& values, std::size_t order,
      const boost::multi_array<double, 4>& reference_values,
      const Eigen::Ref<const EigenRowArrayXXd> X,
      const boost::multi_array<double, 3>& J,
      const Eigen::Ref<const EigenArrayXd> detJ,
      const boost::multi_array<double, 3>& K) const
  {
    assert(_ufc_element);
    std::size_t num_points = X.rows();
    _ufc_element->transform_reference_basis_derivatives(
        values.data(), order, num_points, reference_values.data(), X.data(),
        J.data(), detJ.data(), K.data(), 1);
  }

  /// Evaluate all basis functions at given point in cell
  void evaluate_basis_all(double* values, const double* x,
                          const double* coordinate_dofs,
                          int cell_orientation) const
  {
    dolfin_assert(_ufc_element);
    _ufc_element->evaluate_basis_all(values, x, coordinate_dofs,
                                     cell_orientation);
  }

  /// Evaluate order n derivatives of basis function i at given point in cell
  // void evaluate_basis_derivatives(std::uint32_t i, std::uint32_t n,
  //                                 double* values, const double* x,
  //                                 const double* coordinate_dofs,
  //                                 int cell_orientation) const
  // {
  //   dolfin_assert(_ufc_element);
  //   _ufc_element->evaluate_basis_derivatives(i, n, values, x,
  //   coordinate_dofs,
  //                                            cell_orientation);
  // }

  /// Evaluate order n derivatives of all basis functions at given
  /// point in cell
  // void evaluate_basis_derivatives_all(std::uint32_t n, double* values,
  //                                     const double* x,
  //                                     const double* coordinate_dofs,
  //                                     int cell_orientation) const
  // {
  //   dolfin_assert(_ufc_element);
  //   _ufc_element->evaluate_basis_derivatives_all(n, values, x,
  //   coordinate_dofs,
  //                                                cell_orientation);
  // }

  /// Tabulate the coordinates of all dofs on an element
  ///
  /// @param[in,out]    coordinates (boost::multi_array<double, 2>)
  ///         The coordinates of all dofs on a cell.
  /// @param[in]    coordinate_dofs (std::vector<double>)
  ///         The cell coordinates
  /// @param[in]    cell (Cell)
  ///         The cell.
  void tabulate_dof_coordinates(boost::multi_array<double, 2>& coordinates,
                                const std::vector<double>& coordinate_dofs,
                                const mesh::Cell& cell) const;

  /// Return the number of sub elements (for a mixed element)
  /// @return std::size_t
  ///   number of sub-elements
  std::size_t num_sub_elements() const
  {
    dolfin_assert(_ufc_element);
    return _ufc_element->num_sub_elements();
  }

  //--- DOLFIN-specific extensions of the interface ---

  /// Return simple hash of the signature string
  std::size_t hash() const { return _hash; }

  /// Create a new finite element for sub element i (for a mixed
  /// element)
  std::shared_ptr<FiniteElement> create_sub_element(std::size_t i) const
  {
    dolfin_assert(_ufc_element);
    std::shared_ptr<ufc::finite_element> ufc_element(
        _ufc_element->create_sub_element(i));
    return std::make_shared<FiniteElement>(ufc_element);
  }

  /// Create a new class instance
  std::shared_ptr<FiniteElement> create() const
  {
    dolfin_assert(_ufc_element);
    std::shared_ptr<ufc::finite_element> ufc_element(_ufc_element->create());
    return std::make_shared<FiniteElement>(ufc_element);
  }

  /// Extract sub finite element for component
  std::shared_ptr<FiniteElement>
  extract_sub_element(const std::vector<std::size_t>& component) const;

  /// Return underlying UFC element. Intended for libray usage only
  /// and may change.
  std::shared_ptr<const ufc::finite_element> ufc_element() const
  {
    return _ufc_element;
  }

private:
  // UFC finite element
  std::shared_ptr<const ufc::finite_element> _ufc_element;

  // Recursively extract sub finite element
  static std::shared_ptr<FiniteElement>
  extract_sub_element(const FiniteElement& finite_element,
                      const std::vector<std::size_t>& component);

  // Simple hash of the signature string
  std::size_t _hash;
};
}
}
