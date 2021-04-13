// Copyright (C) 2018-2021 Garth N. Wells, Jørgen S. Dokken, Igor A. Baratta
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateElement.h"
#include <Eigen/Dense>
#include <basix.h>
#include <basix/finite-element.h>
#include <dolfinx/mesh/cell_types.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
CoordinateElement::CoordinateElement(
    std::shared_ptr<basix::FiniteElement> element, int geometric_dimension,
    const std::string& signature, const ElementDofLayout& dof_layout)
    : _gdim(geometric_dimension), _signature(signature),
      _dof_layout(dof_layout), _element(element)
{
  int degree = _element->degree();
  const char* cell_type
      = basix::cell::type_to_str(_element->cell_type()).c_str();

  const char* family_name
      = basix::element::type_to_str(_element->family()).c_str();

  _basix_element_handle
      = basix::register_element(family_name, cell_type, degree);

  const mesh::CellType cell = cell_shape();
  _is_affine = mesh::is_simplex(cell) and degree == 1;
}
//-----------------------------------------------------------------------------
mesh::CellType CoordinateElement::cell_shape() const
{
  // TODO
  const std::string cell = basix::cell::type_to_str(_element->cell_type());

  const std::map<std::string, mesh::CellType> str_to_type
      = {{"interval", mesh::CellType::interval},
         {"triangle", mesh::CellType::triangle},
         {"quadrilateral", mesh::CellType::quadrilateral},
         {"tetrahedron", mesh::CellType::tetrahedron},
         {"hexahedron", mesh::CellType::hexahedron}};

  auto it = str_to_type.find(cell);
  if (it == str_to_type.end())
    throw std::runtime_error("Problem with cell type");
  return it->second;
}
//-----------------------------------------------------------------------------
int CoordinateElement::topological_dimension() const
{
  return basix::cell::topology(_element->cell_type()).size() - 1;
}
//-----------------------------------------------------------------------------
int CoordinateElement::geometric_dimension() const { return _gdim; }
//-----------------------------------------------------------------------------
const ElementDofLayout& CoordinateElement::dof_layout() const
{
  return _dof_layout;
}
//-----------------------------------------------------------------------------
void CoordinateElement::push_forward(array2d<double>& x,
                                     const array2d<double>& cell_geometry,
                                     const xt::xtensor<double, 2>& phi) const
{
  assert((int)x.shape[1] == this->geometric_dimension());
  assert(phi.shape(2) == cell_geometry.shape[0]);

  // Compute physical coordinates
  // x = phi * cell_geometry;
  std::fill(x.data(), x.data() + x.size(), 0.0);
  for (std::size_t i = 0; i < x.shape[0]; ++i)
    for (std::size_t j = 0; j < x.shape[1]; ++j)
      for (std::size_t k = 0; k < cell_geometry.shape[0]; ++k)
        x(i, j) += phi(i, k) * cell_geometry(k, j);
}
//-----------------------------------------------------------------------------
void CoordinateElement::compute_reference_geometry(
    array2d<double>& X, std::vector<double>& J, tcb::span<double> detJ,
    std::vector<double>& K, const array2d<double>& x,
    const array2d<double>& cell_geometry) const
{
  // Number of points
  int num_points = x.shape[0];
  if (num_points == 0)
    return;

  Eigen::Map<
      Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
      _X(X.data(), X.shape[0], X.shape[1]);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _x(x.data(), x.shape[0], x.shape[1]);
  Eigen::Map<const Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                                Eigen::RowMajor>>
      _cell_geometry(cell_geometry.data(), cell_geometry.shape[0],
                     cell_geometry.shape[1]);

  // in-argument checks
  const int tdim = this->topological_dimension();
  const int gdim = this->geometric_dimension();
  assert(_x.cols() == gdim);
  assert(_cell_geometry.cols() == gdim);

  // In/out size checks
  assert(_X.rows() == num_points);
  assert(_X.cols() == tdim);
  assert((int)J.size() == num_points * gdim * tdim);
  assert((int)detJ.size() == num_points);
  assert((int)K.size() == num_points * gdim * tdim);

  // FIXME: Array and matrix rows/cols transpose etc all very tortuous
  // FIXME: tidy up and sort out

  const int d = _cell_geometry.rows();
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dphi(
      d, tdim);

  Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      tabulated_data(tdim + 1, _cell_geometry.rows());

  if (_is_affine)
  {
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> x0(_gdim);
    Eigen::ArrayXXd X0 = Eigen::ArrayXXd::Zero(1, tdim);

    basix::tabulate(_basix_element_handle, tabulated_data.data(), 1, X0.data(),
                    1);

    // Compute physical coordinates at X=0.
    x0 = tabulated_data.row(0).matrix() * _cell_geometry.matrix();

    // Compute Jacobian and inverse
    dphi = tabulated_data.block(1, 0, tdim, d).transpose();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 3, 3>
        J0(gdim, tdim);
    J0 = _cell_geometry.matrix().transpose() * dphi;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 3, 3>
        K0(tdim, gdim);

    // Fill result for J, K and detJ
    if (gdim == tdim)
    {
      K0 = J0.inverse();
      std::fill(detJ.begin(), detJ.end(), J0.determinant());
    }
    else
    {
      // Penrose-Moore pseudo-inverse
      K0 = (J0.transpose() * J0).inverse() * J0.transpose();
      // detJ.fill(std::sqrt((J0.transpose() * J0).determinant()));
      std::fill(detJ.begin(), detJ.end(),
                std::sqrt((J0.transpose() * J0).determinant()));
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
      _X.row(ip) = K0 * (_x.row(ip).matrix().transpose() - x0);
  }
  else
  {
    // Newton's method for non-affine geometry
    Eigen::Matrix<double, Eigen::Dynamic, 1, Eigen::ColMajor, 3, 1> xk(
        _x.cols());
    Eigen::RowVectorXd Xk(tdim);
    Eigen::RowVectorXd dX(tdim);

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
      int k;
      for (k = 0; k < non_affine_max_its; ++k)
      {
        basix::tabulate(_basix_element_handle, tabulated_data.data(), 1,
                        Xk.data(), 1);

        // Compute physical coordinates
        xk = tabulated_data.row(0).matrix() * _cell_geometry.matrix();

        // Compute Jacobian and inverse
        dphi = tabulated_data.block(1, 0, tdim, d).transpose();
        Jview = _cell_geometry.matrix().transpose() * dphi;
        if (gdim == tdim)
          Kview = Jview.inverse();
        else
          // Penrose-Moore pseudo-inverse
          Kview = (Jview.transpose() * Jview).inverse() * Jview.transpose();

        // Increment to new point in reference
        dX = Kview * (_x.row(ip).matrix().transpose() - xk);
        if (dX.norm() < non_affine_atol)
          break;
        Xk += dX;
      }
      if (k == non_affine_max_its)
      {
        throw std::runtime_error(
            "Newton method failed to converge for non-affine geometry");
      }
      _X.row(ip) = Xk;
      if (gdim == tdim)
        detJ[ip] = Jview.determinant();
      else
        detJ[ip] = std::sqrt((Jview.transpose() * Jview).determinant());
    }
  }
}
//-----------------------------------------------------------------------------
void CoordinateElement::permute_dofs(std::int32_t* dofs,
                                     const uint32_t cell_perm) const
{
  basix::permute_dofs(_basix_element_handle, dofs, cell_perm);
}
//-----------------------------------------------------------------------------
void CoordinateElement::unpermute_dofs(std::int32_t* dofs,
                                       const uint32_t cell_perm) const
{
  basix::unpermute_dofs(_basix_element_handle, dofs, cell_perm);
}
//-----------------------------------------------------------------------------
bool CoordinateElement::needs_permutation_data() const
{
  return !basix::dof_transformations_are_identity(_basix_element_handle);
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 4>
CoordinateElement::tabulate(int n, const array2d<double>& X) const
{
  auto _X = xt::adapt(X.data(), X.shape);
  return _element->tabulate(n, _X);
}
//-----------------------------------------------------------------------------
void CoordinateElement::compute_jacobian_data(
    const xt::xtensor<double, 4>& tabulated_data, const array2d<double>& X,
    const array2d<double>& cell_geometry, std::vector<double>& J,
    tcb::span<double> detJ, std::vector<double>& K) const
{
  // Number of points
  int num_points = X.shape[0];
  if (num_points == 0)
    return;

  // in-argument checks
  const int tdim = this->topological_dimension();
  const int gdim = this->geometric_dimension();
  assert(int(cell_geometry.shape[1]) == gdim);

  // In/out size checks
  assert((int)J.size() == num_points * gdim * tdim);
  const int d = cell_geometry.shape[0];
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> dphi(
      d, tdim);
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      _cell_geometry(cell_geometry.data(), cell_geometry.shape[0],
                     cell_geometry.shape[1]);
  if (_is_affine)
  {
    for (std::int32_t i = 0; i < tdim; ++i)
    {
      auto dphi_i = xt::view(tabulated_data, i + 1, 0, xt::all(), 0);
      for (std::int32_t j = 0; j < d; ++j)
        dphi(j, i) = dphi_i(j);
    }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 3, 3>
        J0 = _cell_geometry.transpose() * dphi;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor, 3, 3>
        K0(tdim, gdim);

    // NOTE: should we use xtensor-blas?
    // Fill result for J, K and detJ
    if (gdim == tdim)
    {
      K0 = J0.inverse();
      std::fill(detJ.begin(), detJ.end(), J0.determinant());
    }
    else
    {
      // Penrose-Moore pseudo-inverse
      K0 = (J0.transpose() * J0).inverse() * J0.transpose();
      std::fill(detJ.begin(), detJ.end(),
                std::sqrt((J0.transpose() * J0).determinant()));
    }

    // As J0 and K0 are constant for affine meshes, replicate per intepolation
    // point
    for (int ip = 0; ip < num_points; ip++)
    {
      std::copy_n(J0.data(), J0.size(), J.data() + ip * J0.size());
      std::copy_n(K0.data(), K0.size(), K.data() + ip * K0.size());
    }
  }
  else
  {
    for (int ip = 0; ip < num_points; ++ip)
    {
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          Jview(J.data() + ip * gdim * tdim, gdim, tdim);
      Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                               Eigen::RowMajor>>
          Kview(K.data() + ip * gdim * tdim, tdim, gdim);
      for (std::int32_t i = 0; i < tdim; ++i)
      {
        auto dphi_i = xt::view(tabulated_data, i + 1, ip, xt::all(), 0);
        for (std::int32_t j = 0; j < d; ++j)
          dphi(j, i) = dphi_i[j];
      }
      Jview = _cell_geometry.transpose() * dphi;
      if (gdim == tdim)
      {
        Kview = Jview.inverse();
        detJ[ip] = Jview.determinant();
      }
      else
      {
        // Penrose-Moore pseudo-inverse
        Kview = (Jview.transpose() * Jview).inverse() * Jview.transpose();
        detJ[ip] = std::sqrt((Jview.transpose() * Jview).determinant());
      }
    }
  }
}
//--------------------------------------------------------------------------------
void CoordinateElement::compute_jacobian(
    const xt::xtensor<double, 4>& tabulated_data,
    const xt::xtensor<double, 2>& cell_geometry,
    xt::xtensor<double, 3>& J) const
{
  // Number of points
  int num_points = tabulated_data.shape(1);
  if (num_points == 0)
    return;

  assert(int(J.shape(0)) == num_points);

  // in-argument checks
  const int tdim = this->topological_dimension();
  const int gdim = this->geometric_dimension();
  assert(int(cell_geometry.shape(1)) == gdim);

  // In/out size checks
  assert((int)J.size() == num_points * gdim * tdim);
  const int d = cell_geometry.shape(0);
  xt::xtensor<double, 2> dphi = xt::empty<double>({tdim, d});

  if (_is_affine)
  {
    dphi = xt::view(tabulated_data, xt::range(1, tdim + 1), 0, xt::all(), 0);
    auto J0 = xt::linalg::dot(dphi, cell_geometry);
    J = xt::broadcast(xt::transpose(J0), J.shape());
  }
  else
  {
    for (int ip = 0; ip < num_points; ++ip)
    {
      auto J_ip = xt::view(J, ip, xt::all(), xt::all());
      dphi = xt::view(tabulated_data, xt::range(1, tdim + 1), ip, xt::all(), 0);
      auto J0 = xt::linalg::dot(dphi, cell_geometry);
      J_ip.assign(J0);
    }
  }
}
//--------------------------------------------------------------------------------
void CoordinateElement::compute_jacobian_inverse(
    const xt::xtensor<double, 3>& J, xt::xtensor<double, 3>& K) const
{
  assert(J.shape(0) == K.shape(0));
  assert(J.shape(1) == K.shape(2));
  assert(J.shape(2) == K.shape(1));

  int num_points = J.shape(0);
  const int gdim = J.shape(1);
  const int tdim = K.shape(2);

  xt::xtensor<double, 2> K0 = xt::empty<double>({tdim, gdim});
  xt::xtensor<double, 2> J0 = xt::empty<double>({gdim, tdim});

  if (_is_affine)
  {
    J0 = xt::view(J, 0, xt::all(), xt::all());
    if (gdim == tdim)
      K0 = xt::linalg::inv(J0);
    else
      K0 = xt::linalg::pinv(J0);
    K = xt::broadcast(K0, K.shape());
  }
  else
  {
    for (int ip = 0; ip < num_points; ip++)
    {
      J0 = xt::view(J, ip, xt::all(), xt::all());
      if (gdim == tdim)
        K0 = xt::linalg::inv(J0);
      else
        K0 = xt::linalg::pinv(J0);
      auto K_ip = xt::view(K, ip, xt::all(), xt::all());
      K_ip.assign(K0);
    }
  }
}

//--------------------------------------------------------------------------------
void CoordinateElement::compute_jacobian_determinant(
    const xt::xtensor<double, 3>& J, xt::xtensor<double, 1>& Jdet) const
{
  assert(J.shape(0) == Jdet.shape(0));
  int num_points = J.shape(0);

  if (_is_affine)
  {
    auto J0 = xt::view(J, 0, xt::all(), xt::all());
    Jdet.fill(xt::linalg::det(J0));
  }
  else
  {
    for (int ip = 0; ip < num_points; ip++)
    {
      auto Jip = xt::view(J, ip, xt::all(), xt::all());
      Jdet[ip] = xt::linalg::det(Jip);
    }
  }
}