// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateMapping.h"
#include <dolfin/common/types.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
CoordinateMapping::CoordinateMapping(
    mesh::CellType cell_type, int topological_dimension,
    int geometric_dimension, std::string signature,
    std::function<void(double*, int, const double*, const double*)>
        compute_physical_coordinates,
    std::function<void(double*, double*, double*, double*, int, const double*,
                       const double*, int)>
        compute_reference_geometry)
    : _tdim(topological_dimension), _gdim(geometric_dimension),
      _cell(cell_type), _signature(signature),
      _compute_physical_coordinates(compute_physical_coordinates),
      _compute_reference_geometry(compute_reference_geometry)
{
}
//-----------------------------------------------------------------------------
std::string CoordinateMapping::signature() const { return _signature; }
//-----------------------------------------------------------------------------
mesh::CellType CoordinateMapping::cell_shape() const { return _cell; }
//-----------------------------------------------------------------------------
std::uint32_t CoordinateMapping::topological_dimension() const { return _tdim; }
//-----------------------------------------------------------------------------
std::uint32_t CoordinateMapping::geometric_dimension() const { return _gdim; }
//-----------------------------------------------------------------------------
void CoordinateMapping::compute_physical_coordinates(
    Eigen::Ref<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        x,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& coordinate_dofs)
    const
{
  assert(_compute_physical_coordinates);
  assert(x.rows() == X.rows());
  assert(x.cols() == _gdim);
  assert(X.cols() == _tdim);
  _compute_physical_coordinates(x.data(), X.rows(), X.data(),
                                coordinate_dofs.data());
}
//-----------------------------------------------------------------------------
void CoordinateMapping::compute_reference_geometry(
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X,
    Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
    Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1>> detJ,
    Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& coordinate_dofs)
    const
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
                              num_points, x.data(), coordinate_dofs.data(), 1);
}
//-----------------------------------------------------------------------------
