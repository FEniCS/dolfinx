// Copyright (C) 2018 Garth N. Wells
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
  /// @param[in] cell_type
  /// @param[in] topological_dimension
  /// @param[in] geometric_dimension
  /// @param[in] signature
  /// @param[in] dof_layout Layout of the geometry degrees-of-freedom
  /// @param[in] compute_physical_coordinates Push-forward function from
  ///   reference to physical coordinates
  /// @param[in] compute_reference_geometry Pull-back function from
  ///   physical coordinates to reference coordinates
  CoordinateElement(
      mesh::CellType cell_type, int topological_dimension,
      int geometric_dimension, const std::string& signature,
      const ElementDofLayout& dof_layout,
      std::function<void(double*, int, const double*, const double*)>
          compute_physical_coordinates,
      std::function<void(double*, double*, double*, double*, int, const double*,
                         const double*)>
          compute_reference_geometry);

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
          cell_geometry) const;

private:
  int _tdim, _gdim;

  mesh::CellType _cell;

  std::string _signature;

  ElementDofLayout _dof_layout;

  std::function<void(double*, int, const double*, const double*)>
      _compute_physical_coordinates;

  std::function<void(double*, double*, double*, double*, int, const double*,
                     const double*)>
      _compute_reference_geometry;
};
} // namespace dolfinx::fem
