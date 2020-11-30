// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateElement.h"
#include <libtab.h>
#include <unsupported/Eigen/CXX11/Tensor>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
CoordinateElement::CoordinateElement(
    const libtab::FiniteElement& libtab_element, int geometric_dimension,
    const std::string& signature, const ElementDofLayout& dof_layout)
    : _gdim(geometric_dimension), _signature(signature),
      _dof_layout(dof_layout),
      _libtab_element(
          std::make_shared<const libtab::FiniteElement>(libtab_element))
{
  const mesh::CellType cell = cell_shape();
  _is_affine
      = ((cell == mesh::CellType::interval or cell == mesh::CellType::triangle
          or cell == mesh::CellType::tetrahedron)
         and _libtab_element->degree() == 1);
}
//-----------------------------------------------------------------------------
std::string CoordinateElement::signature() const { return _signature; }
//-----------------------------------------------------------------------------
mesh::CellType CoordinateElement::cell_shape() const
{
  switch (_libtab_element->cell_type())
  {
  case libtab::cell::type::interval:
    return mesh::CellType::interval;
  case libtab::cell::type::triangle:
    return mesh::CellType::triangle;
  case libtab::cell::type::quadrilateral:
    return mesh::CellType::quadrilateral;
  case libtab::cell::type::hexahedron:
    return mesh::CellType::hexahedron;
  case libtab::cell::type::tetrahedron:
    return mesh::CellType::tetrahedron;
  default:
    throw std::runtime_error("Invalid cell shape in CoordinateElement");
  }
}
//-----------------------------------------------------------------------------
int CoordinateElement::topological_dimension() const
{
  return libtab::cell::topological_dimension(_libtab_element->cell_type());
}
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
  assert(x.cols() == this->geometric_dimension());
  assert(X.cols() == this->topological_dimension());

  // Compute physical coordinates

  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> phi
      = _libtab_element->tabulate(0, X)[0];

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
                                        Eigen::RowMajor>>& cell_geometry,
    double eps) const
{
  // Number of points
  int num_points = x.rows();
  if (num_points == 0)
    return;

  // in-argument checks
  const int tdim = this->topological_dimension();
  const int gdim = this->geometric_dimension();
  assert(x.cols() == gdim);
  assert(cell_geometry.cols() == gdim);

  // In/out size checks
  assert(X.rows() == num_points);
  assert(X.cols() == tdim);
  assert(J.dimension(0) == num_points);
  assert(J.dimension(1) == gdim);
  assert(J.dimension(2) == tdim);
  assert(detJ.rows() == num_points);
  assert(K.dimension(0) == num_points);
  assert(K.dimension(1) == tdim);
  assert(K.dimension(2) == gdim);

  // FIXME: Array and matrix rows/cols transpose etc all very tortuous
  // FIXME: tidy up and sort out

  const int d = cell_geometry.rows();
  Eigen::VectorXd phi(d);
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dphi(
      d, tdim);

  std::vector<Eigen::ArrayXXd> tabulated_data;

  if (_is_affine)
  {
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> x0(_gdim);
    Eigen::ArrayXXd X0 = Eigen::ArrayXXd::Zero(1, tdim);

    tabulated_data = _libtab_element->tabulate(1, X0);

    // Compute physical coordinates at X=0.
    phi = tabulated_data[0].transpose();
    x0 = cell_geometry.matrix().transpose() * phi;

    // Compute Jacobian and inverse
    for (std::size_t dim = 0; dim + 1 < tabulated_data.size(); ++dim)
      dphi.col(dim) = tabulated_data[dim + 1].row(0);
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 3, 3>
        J0(gdim, tdim);
    J0 = cell_geometry.matrix().transpose() * dphi;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 3, 3>
        K0(tdim, gdim);

    // Fill result for J, K and detJ
    if (gdim == tdim)
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

    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Jview(J.data(), gdim * num_points, tdim);
    Jview = J0.replicate(num_points, 1);
    Eigen::Map<
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
        Kview(K.data(), tdim * num_points, gdim);
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
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> Xk(tdim, 1);

    for (int ip = 0; ip < num_points; ++ip)
    {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          Jview(J.data() + ip * gdim * tdim, gdim, tdim);
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          Kview(K.data() + ip * gdim * tdim, tdim, gdim);
      // TODO: Xk - use cell midpoint instead?
      Xk.setZero();
      const int max_its = 10;
      int k;
      for (k = 0; k < max_its; ++k)
      {

        tabulated_data = _libtab_element->tabulate(1, Xk);

        // Compute physical coordinates
        phi = tabulated_data[0].transpose();
        xk = cell_geometry.matrix().transpose() * phi;

        // Compute Jacobian and inverse
        for (std::size_t dim = 0; dim + 1 < tabulated_data.size(); ++dim)
          dphi.col(dim) = tabulated_data[dim + 1].row(0);
        Jview = cell_geometry.matrix().transpose() * dphi;
        if (gdim == tdim)
          Kview = Jview.inverse();
        else
          // Penrose-Moore pseudo-inverse
          Kview = (Jview.transpose() * Jview).inverse() * Jview.transpose();

        // Increment to new point in reference
        Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> dX
            = Kview * (x.row(ip).matrix().transpose() - xk);
        if (dX.squaredNorm() < eps)
          break;
        Xk += dX;
      }
      if (k == max_its)
      {
        throw std::runtime_error(
            "Newton method failed to converge for non-affine geometry");
      }
      X.row(ip) = Xk;
      if (gdim == tdim)
        detJ.row(ip) = Jview.determinant();
      else
        detJ.row(ip) = std::sqrt((Jview.transpose() * Jview).determinant());
    }
  }
}
//-----------------------------------------------------------------------------
