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
    const ElementDofLayout& dof_layout, bool is_affine,
    const std::function<int(double*, int, int, const double*)>&
        evaluate_basis_derivatives)
    : _tdim(topological_dimension), _gdim(geometric_dimension),
      _cell(cell_type), _signature(signature), _dof_layout(dof_layout),
      _is_affine(is_affine),
      _evaluate_basis_derivatives(evaluate_basis_derivatives)
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
  assert(x.rows() == X.rows());
  assert(x.cols() == _gdim);
  assert(X.cols() == _tdim);

  // Compute physical coordinates
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> phi(
      X.rows(), cell_geometry.rows());

  _evaluate_basis_derivatives(phi.data(), 0, X.rows(), X.data());
  x = phi * cell_geometry.matrix();
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
  // Number of points
  int num_points = x.rows();
  if (num_points == 0)
    return;

  // in-argument checks
  assert(x.cols() == this->geometric_dimension());
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

  const int d = cell_geometry.rows();
  Eigen::VectorXd phi(d);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dphi(
      d, _tdim);

  if (_is_affine)
  {
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> x0(_gdim);
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> X0(_tdim);
    X0.setZero();
    // Compute physical coordinates at X=0.
    _evaluate_basis_derivatives(phi.data(), 0, 1, X0.data());
    x0 = cell_geometry.matrix().transpose() * phi;

    // Compute Jacobian and inverse
    _evaluate_basis_derivatives(dphi.data(), 1, 1, X0.data());
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 3, 3>
        J0(_gdim, _tdim);
    J0 = cell_geometry.matrix().transpose() * dphi;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 3, 3>
        K0(_tdim, _gdim);
    if (_gdim == _tdim)
    {
      K0 = J0.inverse();
      detJ.fill(J0.determinant());
    }
    else
    {
      // Penrose-Moore pseudo-inverse
      K0 = (J0.transpose() * J0).inverse() * J0.transpose();
      detJ.fill(std::sqrt((J0.transpose() * J0).determinant()));
    }

    // Fill result for J, K and detJ
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Jview(J.data(), _gdim * num_points, _tdim);
    Jview = J0.replicate(num_points, 1);
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Kview(K.data(), _tdim * num_points, _gdim);
    Kview = K0.replicate(num_points, 1);

    // Calculate X for each point
    for (int ip = 0; ip < num_points; ++ip)
      X.row(ip) = K0 * (x.row(ip).matrix().transpose() - x0);
  }
  else
  {
    // Newton's method for non-affine geometry
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> xk(x.cols(),
                                                                       1);
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> Xk(_tdim,
                                                                       1);

    for (int ip = 0; ip < num_points; ++ip)
    {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          Jview(J.data() + ip * _gdim * _tdim, _gdim, _tdim);
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          Kview(K.data() + ip * _gdim * _tdim, _tdim, _gdim);
      // TODO: Xk - use cell midpoint instead?
      Xk.setZero();
      const int max_its = 10;
      int k;
      for (k = 0; k < max_its; ++k)
      {
        // Compute physical coordinates
        _evaluate_basis_derivatives(phi.data(), 0, 1, Xk.data());
        xk = cell_geometry.matrix().transpose() * phi;

        // Compute Jacobian and inverse
        _evaluate_basis_derivatives(dphi.data(), 1, 1, Xk.data());
        Jview = cell_geometry.matrix().transpose() * dphi;
        if (_gdim == _tdim)
          Kview = Jview.inverse();
        else
          // Penrose-Moore pseudo-inverse
          Kview = (Jview.transpose() * Jview).inverse() * Jview.transpose();

        // Increment to new point in reference
        Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> dX
            = Kview * (x.row(ip).matrix().transpose() - xk);
        if (dX.squaredNorm() < 1e-12)
          break;
        Xk += dX;
      }
      if (k == max_its)
      {
        throw std::runtime_error(
            "Newton method failed to converge for non-affine geometry");
      }
      X.row(ip) = Xk;
      if (_gdim == _tdim)
        detJ.row(ip) = Jview.determinant();
      else
        detJ.row(ip) = std::sqrt((Jview.transpose() * Jview).determinant());
    }
  }
}
//-----------------------------------------------------------------------------
