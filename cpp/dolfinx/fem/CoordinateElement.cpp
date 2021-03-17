// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateElement.h"
#include <Eigen/Dense>
#include <basix.h>

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
CoordinateElement::CoordinateElement(
    int basix_element_handle, int geometric_dimension,
    const std::string& signature, const ElementDofLayout& dof_layout,
    bool needs_permutation_data,
    std::function<int(int*, const uint32_t)> permute_dofs,
    std::function<int(int*, const uint32_t)> unpermute_dofs)
    : _gdim(geometric_dimension), _signature(signature),
      _dof_layout(dof_layout), _basix_element_handle(basix_element_handle),
      _needs_permutation_data(needs_permutation_data),
      _permute_dofs(permute_dofs), _unpermute_dofs(unpermute_dofs)
{
  const mesh::CellType cell = cell_shape();
  int degree = basix::degree(basix_element_handle);
  _is_affine
      = ((cell == mesh::CellType::interval or cell == mesh::CellType::triangle
          or cell == mesh::CellType::tetrahedron)
         and degree == 1);
}
//-----------------------------------------------------------------------------
std::string CoordinateElement::signature() const { return _signature; }
//-----------------------------------------------------------------------------
mesh::CellType CoordinateElement::cell_shape() const
{
  // TODO
  const std::string cell = basix::cell_type(_basix_element_handle);

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
  const std::string cell = basix::cell_type(_basix_element_handle);
  return basix::topology(cell.c_str()).size() - 1;
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
                                     const array2d<double>& phi) const
{
  assert((int)x.shape[1] == this->geometric_dimension());
  assert((int)phi.shape[1] == this->geometric_dimension());

  // Compute physical coordinates
  // x = phi * cell_geometry;
  std::fill(x.data(), x.data() + x.size(), 0.0);
  for (std::size_t i = 0; i < x.shape[0]; ++i)
    for (std::size_t j = 0; j < x.shape[1]; ++j)
      for (std::size_t k = 0; k < cell_geometry.shape[0]; ++k)
        x(i, j) += phi(k, i) * cell_geometry(k, j);
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
void CoordinateElement::permute_dofs(int* dofs, const uint32_t cell_perm) const
{
  _permute_dofs(dofs, cell_perm);
}
//-----------------------------------------------------------------------------
void CoordinateElement::unpermute_dofs(int* dofs,
                                       const uint32_t cell_perm) const
{
  _unpermute_dofs(dofs, cell_perm);
}
//-----------------------------------------------------------------------------
bool CoordinateElement::needs_permutation_data() const
{
  return _needs_permutation_data;
}
//-----------------------------------------------------------------------------
void CoordinateElement::tabulate_shape_functions(const array2d<double>& X,
                                                 array2d<double>& phi) const
{
  assert(phi.shape[0] == X.shape[0]);
  assert((int)phi.shape[1] == _gdim);

  // std::vector<double> data(30);
  basix::tabulate(_basix_element_handle, phi.data(), 0, X.data(), X.shape[0]);
}
