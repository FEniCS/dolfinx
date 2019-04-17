// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ReferenceCellTopology.h"
#include <Eigen/Dense>
#include <cstdint>
#include <dolfin/mesh/CellType.h>
#include <functional>
#include <string>
#include <unsupported/Eigen/CXX11/Tensor>

struct ufc_coordinate_mapping;

namespace dolfin
{

namespace fem
{

/// This class manages coordinate mappings for isoparametric cells.

class CoordinateMapping
{
public:
  /// Create coordinate mapping from UFC coordinate mapping
  /// @param cm (ufc::coordinate_mapping)
  ///  UFC coordinate mapping
  CoordinateMapping(const ufc_coordinate_mapping& cm);

  CoordinateMapping(
      CellType cell_type, int topological_dimension, int geometric_dimension,
      std::string signature,
      std::function<void(double*, int, const double*, const double*)>
          compute_physical_coordinates,
      std::function<void(double*, double*, double*, double*, int, const double*,
                         const double*, int)>
          compute_reference_geometry);

  /// Destructor
  virtual ~CoordinateMapping() = default;

  //--- Direct wrappers for ufc_coordinate_mapping ---

  /// Return a string identifying the finite element
  /// @return std::string
  std::string signature() const;

  /// Return the cell shape
  /// @return CellType
  CellType cell_shape() const;

  /// Return the topological dimension of the cell shape
  /// @return std::size_t
  std::uint32_t topological_dimension() const;

  /// Return the geometric dimension of the cell shape
  /// @return std::uint32_t
  std::uint32_t geometric_dimension() const;

  /// Compute physical coordinates x for points X  in the reference
  /// configuration
  void compute_physical_coordinates(
      Eigen::Ref<
          Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          x,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& X,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          coordinate_dofs) const;

  /// Compute reference coordinates X, and J, detJ and K for physical
  /// coordinates x
  void compute_reference_geometry(
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X,
      Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      Eigen::Array<double, Eigen::Dynamic, 1>& detJ,
      Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>& x,
      const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic,
                                          Eigen::Dynamic, Eigen::RowMajor>>&
          coordinate_dofs) const;

private:
  int _tdim, _gdim;

  CellType _cell;

  std::string _signature;

  std::function<void(double*, int, const double*, const double*)>
      _compute_physical_coordinates;

  std::function<void(double*, double*, double*, double*, int, const double*,
                     const double*, int)>
      _compute_reference_geometry;
};
} // namespace fem
} // namespace dolfin
