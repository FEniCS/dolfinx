// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateElement.h"
#include "FiniteElement.h"
#include <iostream>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
CoordinateElement::CoordinateElement(
    mesh::CellType cell_type, int topological_dimension,
    int geometric_dimension, const std::string& signature,
    const ElementDofLayout& dof_layout, Eigen::Vector3d reference_midpoint,
    std::shared_ptr<const FiniteElement> element)
    : _tdim(topological_dimension), _gdim(geometric_dimension),
      _cell(cell_type), _signature(signature), _dof_layout(dof_layout),
      _reference_midpoint(reference_midpoint), _finite_element(element)
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
  assert(_finite_element);
  assert(x.rows() == X.rows());
  assert(x.cols() == _gdim);
  assert(X.cols() == _tdim);

  // Compute physical coordinates
  Eigen::Tensor<double, 3, Eigen::RowMajor> phi(1, X.rows(),
                                                cell_geometry.rows());
  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      phi_mat(phi.data(), X.rows(), cell_geometry.rows());

  _finite_element->evaluate_reference_basis(phi, X);
  x = phi_mat * cell_geometry.matrix();
}
//-----------------------------------------------------------------------------
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
  assert(_finite_element);

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

  // Newton's method
  Eigen::VectorXd xk(x.cols());
  const int d = cell_geometry.rows();

  Eigen::Tensor<double, 3, Eigen::RowMajor> phi_tensor(1, 1, d);
  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      phi(phi_tensor.data(), d, 1);

  Eigen::Tensor<double, 4, Eigen::RowMajor> dphi_tensor(1, 1, d, _gdim);
  Eigen::Map<
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      dphi(dphi_tensor.data(), d, _gdim);

  for (int ip = 0; ip < num_points; ++ip)
  {
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Jview(J.data() + ip * _gdim * _tdim, _gdim, _tdim);
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Kview(K.data() + ip * _gdim * _tdim, _tdim, _gdim);
    Eigen::RowVectorXd Xk(_tdim);
    // Xk = _reference_midpoint.head(_tdim);
    // or Xk.setZero() ?
    Xk.setZero();
    const int max_its = 10;
    int k;
    for (k = 0; k < max_its; ++k)
    {
      // Compute physical coordinates
      _finite_element->evaluate_reference_basis(phi_tensor, Xk);
      xk = cell_geometry.matrix().transpose() * phi;

      // Compute Jacobian and inverse
      _finite_element->evaluate_reference_basis_derivatives(dphi_tensor, 1, Xk);
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
