// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateElement.h"
#include <unsupported/Eigen/CXX11/Tensor>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
CoordinateElement::CoordinateElement(
    mesh::CellType cell_type, int topological_dimension,
    int geometric_dimension, const std::string& signature,
    const ElementDofLayout& dof_layout,
    std::function<void(double*, int, const double*, const double*)>
        compute_physical_coordinates,
    std::function<void(double*, double*, double*, double*, int, const double*,
                       const double*)>
        compute_reference_geometry,
    std::function<int(double*, int, const double*)> evaluate_reference_basis,
    std::function<int(double*, int, int, const double*)>
        evaluate_reference_basis_derivatives)
    : _tdim(topological_dimension), _gdim(geometric_dimension),
      _cell(cell_type), _signature(signature), _dof_layout(dof_layout),
      _compute_physical_coordinates(compute_physical_coordinates),
      _compute_reference_geometry(compute_reference_geometry),
      _evaluate_reference_basis(evaluate_reference_basis),
      _evaluate_reference_basis_derivatives(
          evaluate_reference_basis_derivatives)

{
}
//-----------------------------------------------------------------------------
std::string CoordinateElement::signature() const { return _signature; }
//-----------------------------------------------------------------------------
mesh::CellType CoordinateElement::cell_shape() const { return _cell; }
//-----------------------------------------------------------------------------
int CoordinateElement::topological_dimension() const { return _tdim; }
//-----------------------------------------------------------------------------
int CoordinateElement::geometric_dimension() const { return _gdim; }
//-----------------------------------------------------------------------------
const ElementDofLayout& CoordinateElement::dof_layout() const
{
  return _dof_layout;
}
//-----------------------------------------------------------------------------
void CoordinateElement::push_forward(
    Eigen::Ref<
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        x,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& X,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& cell_geometry) const
{
  assert(_compute_physical_coordinates);
  assert(x.rows() == X.rows());
  assert(x.cols() == _gdim);
  assert(X.cols() == _tdim);
  _compute_physical_coordinates(x.data(), X.rows(), X.data(),
                                cell_geometry.data());
}
//-----------------------------------------------------------------------------

#include <iostream>

void CoordinateElement::compute_reference_geometry(
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& X,
    Eigen::Tensor<double, 3, Eigen::RowMajor>& J,
    Eigen::Ref<Eigen::Array<double, Eigen::Dynamic, 1>> detJ,
    Eigen::Tensor<double, 3, Eigen::RowMajor>& K,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& x,
    const Eigen::Ref<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                        Eigen::RowMajor>>& cell_geometry) const
{
  // Number of points
  int num_points = x.rows();

  // in-argument checks
  assert(x.cols() == this->geometric_dimension());
  // assert(cell_geometry.rows() == space_dimension);
  assert(cell_geometry.cols() == this->geometric_dimension());

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

  // assert(_compute_reference_geometry);
  // _compute_reference_geometry(X.data(), J.data(), detJ.data(), K.data(),
  //                             num_points, x.data(), cell_geometry.data());

  // Newton's method
  Eigen::VectorXd xk(x.cols());
  const int d = cell_geometry.rows();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dphi(
      d, _gdim);
  Eigen::VectorXd phi(d);
  for (int ip = 0; ip < num_points; ++ip)
  {
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Jview(J.data() + ip * _gdim * _tdim, _gdim, _tdim);
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Kview(K.data() + ip * _gdim * _tdim, _tdim, _gdim);
    Eigen::VectorXd Xk(_tdim);
    Xk.setZero(); // Is it better to use midpoint? - expose X(mid) from ufc?
    const int max_its = 10;
    int k;
    for (k = 0; k < max_its; ++k)
    {
      // Compute physical coordinates
      _evaluate_reference_basis(phi.data(), 1, Xk.data());
      xk = cell_geometry.matrix().transpose() * phi;
      // Compute Jacobian and inverse
      _evaluate_reference_basis_derivatives(dphi.data(), 1, 1, Xk.data());

      Jview = cell_geometry.matrix().transpose() * dphi;
      Kview = Jview.inverse();

      // Increment to new point in reference
      Eigen::VectorXd dX = Kview * (x.row(ip).matrix().transpose() - xk);
      if (dX.squaredNorm() < 1e-12)
        break;
      Xk += dX;
    }
    if (k == max_its)
    {
      throw std::runtime_error(
          "Iterations exceeded in Newton iteration for cmap");
    }
    X.row(ip) = Xk;
    detJ.row(ip) = Jview.determinant();
  }
}
//-----------------------------------------------------------------------------
