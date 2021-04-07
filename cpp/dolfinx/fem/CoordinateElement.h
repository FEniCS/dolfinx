// Copyright (C) 2018-2020 Garth N. Wells and Chris N. Richardson
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ElementDofLayout.h"
#include <cstdint>
#include <dolfinx/common/array2d.h>
#include <dolfinx/common/span.hpp>
#include <dolfinx/mesh/cell_types.h>
#include <functional>
#include <memory>
#include <string>
#include <xtensor/xtensor.hpp>

namespace basix
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
  /// @param[in] element Element from basix
  /// @param[in] geometric_dimension Geometric dimension
  /// @param[in] dof_layout Layout of the geometry degrees-of-freedom
  /// @param[in] needs_permutation_data Indicates whether or not the
  /// element needs permutation data (for higher order elements)
  /// @param[in] permute_dofs Function that permutes the DOF numbering
  /// @param[in] unpermute_dofs Function that reverses a DOF permutation
  CoordinateElement(std::shared_ptr<basix::FiniteElement> element,
                    int geometric_dimension, const ElementDofLayout& dof_layout,
                    bool needs_permutation_data,
                    std::function<int(int*, const uint32_t)> permute_dofs,
                    std::function<int(int*, const uint32_t)> unpermute_dofs);

  /// Destructor
  virtual ~CoordinateElement() = default;

  /// Cell shape
  /// @return The cell shape
  mesh::CellType cell_shape() const;

  /// Return the topological dimension of the cell shape
  int topological_dimension() const;

  /// Return the geometric dimension of the cell shape
  int geometric_dimension() const;

  /// Tabulate shape functions up to n-th order derivative at points X in the
  /// reference geometry
  /// Note: Dynamic allocation.
  xt::xtensor<double, 4>
  tabulate_shape_functions(int n, const array2d<double>& X) const;

  /// Compute J, K and detJ for a cell with given geometry, and the
  /// basis functions and first order derivatives at points X
  void compute_jacobian_data(const xt::xtensor<double, 4>& tabulated_data,
                             const array2d<double>& X,
                             const array2d<double>& cell_geometry,
                             std::vector<double>& J, tcb::span<double> detJ,
                             std::vector<double>& K) const;

  /// Return the dof layout
  const ElementDofLayout& dof_layout() const;

  /// Absolute increment stopping criterium for non-affine Newton solver
  double non_affine_atol = 1.0e-8;

  /// Maximum number of iterations for non-affine Newton solver
  int non_affine_max_its = 10;

  /// Compute physical coordinates x for points X  in the reference
  /// configuration
  /// @param[in,out] x The physical coordinates of the reference points X
  /// @param[in] cell_geometry The cell node coordinates (physical)
  /// @param[in] phi Tabulated basis functions at reference points X
  void push_forward(array2d<double>& x, const array2d<double>& cell_geometry,
                    const xt::xtensor<double, 4>& phi) const;

  /// Compute reference coordinates X, and J, detJ and K for physical
  /// coordinates x
  void compute_reference_geometry(array2d<double>& X, std::vector<double>& J,
                                  tcb::span<double> detJ,
                                  std::vector<double>& K,
                                  const array2d<double>& x,
                                  const array2d<double>& cell_geometry) const;

  /// Permutes a list of DOF numbers on a cell
  void permute_dofs(int* dofs, const uint32_t cell_perm) const;

  /// Reverses a DOF permutation
  void unpermute_dofs(int* dofs, const uint32_t cell_perm) const;

  /// Indicates whether the coordinate map needs permutation data
  /// passing in (for higher order geometries)
  bool needs_permutation_data() const;

private:
  // Geometric dimensions
  int _gdim;

  // Layout of dofs on element
  ElementDofLayout _dof_layout;

  // Flag denoting affine map
  bool _is_affine;

  // Basix element
  int _basix_element_handle;

  // Does the element need permutation data
  bool _needs_permutation_data;

  // Dof permutation maker
  std::function<int(int*, const uint32_t)> _permute_dofs;

  // Dof permutation maker
  std::function<int(int*, const uint32_t)> _unpermute_dofs;

  // Actual Element;
  std::shared_ptr<basix::FiniteElement> _element;
};
} // namespace dolfinx::fem
