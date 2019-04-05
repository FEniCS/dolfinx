// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "ReferenceCellTopology.h"
#include <dolfin/common/types.h>
#include <memory>
#include <ufc.h>
#include <unsupported/Eigen/CXX11/Tensor>
#include <vector>

namespace dolfin
{

namespace fem
{

/// This is a wrapper for a UFC coordinate mapping (ufc_coordinate_mapping).

class CoordinateMapping
{
public:
  /// Create coordinate mapping from UFC coordinate mapping (data may be
  /// shared)
  /// @param cm (ufc::coordinate_mapping)
  ///  UFC coordinate mapping
  CoordinateMapping(const ufc_coordinate_mapping& cm)
      : _tdim(cm.topological_dimension), _gdim(cm.geometric_dimension),
        _signature(cm.signature),
        _compute_physical_coordinates(cm.compute_physical_coordinates),
        _compute_reference_geometry(cm.compute_reference_geometry)
  {
    static const std::map<ufc_shape, CellType> ufc_to_cell
        = {{vertex, CellType::point},
           {interval, CellType::interval},
           {triangle, CellType::triangle},
           {tetrahedron, CellType::tetrahedron},
           {quadrilateral, CellType::quadrilateral},
           {hexahedron, CellType::hexahedron}};
    const auto it = ufc_to_cell.find(cm.cell_shape);
    assert(it != ufc_to_cell.end());

    _cell = it->second;
    assert(_tdim == ReferenceCellTopology::dim(_cell));
  }

  /// Destructor
  virtual ~CoordinateMapping() {}

  //--- Direct wrappers for ufc_coordinate_mapping ---

  /// Return a string identifying the finite element
  /// @return std::string
  std::string signature() const { return _signature; }

  /// Return the cell shape
  /// @return CellType
  CellType cell_shape() const { return _cell; }

  /// Return the topological dimension of the cell shape
  /// @return std::size_t
  std::uint32_t topological_dimension() const { return _tdim; }

  /// Return the geometric dimension of the cell shape
  /// @return std::uint32_t
  std::uint32_t geometric_dimension() const { return _gdim; }

  /// Compute physical coordinates x for points X  in the reference
  /// configuration
  void compute_physical_coordinates(
      Eigen::Ref<EigenRowArrayXXd> x,
      const Eigen::Ref<const EigenRowArrayXXd>& X,
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
  {
    assert(_compute_physical_coordinates);
    assert(x.rows() == X.rows());
    assert(x.cols() == _gdim);
    assert(X.cols() == _tdim);
    _compute_physical_coordinates(x.data(), X.rows(), X.data(),
                                  coordinate_dofs.data());
  }

  /// Compute reference coordinates X, and J, detJ and K for physical
  /// coordinates x
  void compute_reference_geometry(
      EigenRowArrayXXd& X, Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
      EigenArrayXd& detJ, Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
      const Eigen::Ref<const EigenRowArrayXXd>& x,
      const Eigen::Ref<const EigenRowArrayXXd>& coordinate_dofs) const
  {
    // Number of points
    int num_points = x.rows();

    // in-argument checks
    assert(x.cols() == this->geometric_dimension());
    // assert(coordinate_dofs.rows() == space_dimension);
    assert(coordinate_dofs.cols() == this->geometric_dimension());

    // In/out size checks
    assert(X.rows() == num_points);
    assert(X.cols() == this->topological_dimension());
    assert(J.dimension(0) == num_points);
    assert(J.dimension(1) == this->geometric_dimension());
    assert(J.dimension(2) == this->topological_dimension());
    assert(detJ.rows() == num_points);
    assert(K.dimension(0) == num_points);
    assert(K.dimension(1) == this->topological_dimension());
    assert(K.dimension(2) == this->geometric_dimension());

    assert(_compute_reference_geometry);
    _compute_reference_geometry(X.data(), J.data(), detJ.data(), K.data(),
                                num_points, x.data(), coordinate_dofs.data(),
                                1);
  }

private:
  int _tdim;
  int _gdim;

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
