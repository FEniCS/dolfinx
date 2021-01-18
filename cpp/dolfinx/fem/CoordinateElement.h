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

namespace dolfinx::fem
{

// FIXME: A dof layout on a reference cell needs to be defined.
/// This class manages coordinate mappings for isoparametric cells.

class CoordinateElement
{
public:
  /// Create a coordinate element
  /// @param[in] basix_element_handle Element handle from basix
  /// @param[in] geometric_dimension Geometric dimension
  /// @param[in] signature Signature string description of coordinate map
  /// @param[in] dof_layout Layout of the geometry degrees-of-freedom
  /// @param[in] needs_permutation_data Indicates whether or not the element
  /// needs permutation data (for higher order elements)
  /// @param[in] get_dof_permutation Function that permutes the DOF numbering
  /// @param[in] apply_dof_transformation Function that permutes the dof
  /// coordinates
  CoordinateElement(
      int basix_element_handle, int geometric_dimension,
      const std::string& signature, const ElementDofLayout& dof_layout,
      bool needs_permutation_data,
      std::function<int(int*, const uint32_t)> get_dof_permutation,
      const std::function<int(double*, const std::uint32_t, const int)>
          apply_dof_transformation);

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

  /// Apply a dof transformation to some coordinate data
  /// @param[in,out] coords The coordinates to be permuted
  /// @param[in] cell_permutation An int encoding the orientation of the cell's subentities
  /// @param[in] dim The dimension of the coordinates (is 2 for 2D, 3 for 3D)
  int apply_dof_transformation(double* coords, std::uint32_t cell_permutation,
                               int dim) const;

  /// Return the geometric dimension of the cell shape
  int geometric_dimension() const;

  /// Return the dof layout
  const ElementDofLayout& dof_layout() const;

  /// Absolute increment stopping criterium for non-affine Newton solver
  double non_affine_atol = 1.0e-8;

  /// Maximum number of iterations for non-affine Newton solver
  int non_affine_max_its = 10;

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
          cell_geometry) const;

  /// Returns a function that gets the reordering of DOFs on a cell
  void get_dof_permutation(int* dofs, const uint32_t cell_perm) const;

  /// Indicates whether the coordinate map needs permutation data passing in
  /// (for higher order geometries)
  bool needs_permutation_data() const;

private:
  // Geometric dimensions
  int _gdim;

  // Signature, usually from UFC
  std::string _signature;

  // Layout of dofs on element
  ElementDofLayout _dof_layout;

  // Flag denoting affine map
  bool _is_affine;

  // Basix element
  int _basix_element_handle;

  // Does the element need permutation data
  bool _needs_permutation_data;

  // Dof permutation maker
  std::function<int(int*, const uint32_t)> _get_dof_permutation;

  // apply_dof_transformation
  std::function<int(double*, const std::uint32_t, const int)>
      _apply_dof_transformation;
};
} // namespace dolfinx::fem
