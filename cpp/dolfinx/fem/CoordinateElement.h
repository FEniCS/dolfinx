// Copyright (C) 2018-2020 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ElementDofLayout.h"
#include <Eigen/Dense>
#include <cstdint>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

// Forward declaration
namespace libtab
{
class FiniteElement;
}

namespace dolfinx::fem
{

// FIXME: A dof layout on a reference cell needs to be defined.
/// This class manages coordinate mappings for isoparametric cells.

class CoordinateElement
{
public:
  /// Create a coordinate element
  /// @param[in] libtab_element Element from libtab
  /// @param[in] geometric_dimension Geometric dimension
  /// @param[in] signature Signature string description of coordinate map
  /// @param[in] dof_layout Layout of the geometry degrees-of-freedom
  CoordinateElement(const libtab::FiniteElement& libtab_element,
                    int geometric_dimension, const std::string& signature,
                    const ElementDofLayout& dof_layout);

  /// Destructor
  virtual ~CoordinateElement() = default;

  /// String identifying the finite element
  /// @return The signature
  std::string signature() const;

  /// Cell shape
  /// @return The cell shape
  mesh::CellType cell_shape() const;

  /// Return the topological dimension of the cell shape
  int topological_dimension() const;

  /// Return the geometric dimension of the cell shape
  int geometric_dimension() const;

  /// Return the dof layout
  const ElementDofLayout& dof_layout() const;

  /// Compute physical coordinates x for points X  in the reference
  /// configuration
  /// @param[in,out] x The physical coordinates of the reference points X
  /// @param[in] X The coordinates on the reference cells
  /// @param[in] cell_geometry The cell node coordinates (physical)
  void push_forward(
      Eigen::Ref<
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          x,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& X,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          cell_geometry) const;

  /// Compute reference coordinates X, and J, detJ and K for physical
  /// coordinates x
  void compute_reference_geometry(
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X,
      Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1>> detJ,
      Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& x,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          cell_geometry,
      double eps = 1.0e-16) const;

private:
  // Geometric dimensions
  int _gdim;

  // Signature, usually from UFC
  std::string _signature;

  // Layout of dofs on element
  ElementDofLayout _dof_layout;

  // Flag denoting affine map
  bool _is_affine;

  // Libtab element
  std::shared_ptr<const libtab::FiniteElement> _libtab_element;

  // Function to evaluate the basis on the underlying element
  // @param basis_values Returned values
  // @param order
  // @param num_points
  // @param reference points
  // std::function<int(double*, int, int, const double*)>
  //    _evaluate_basis_derivatives;
};
} // namespace dolfinx::fem
