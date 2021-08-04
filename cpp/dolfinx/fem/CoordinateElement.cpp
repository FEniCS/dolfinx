// Copyright (C) 2018-2021 Garth N. Wells, Jørgen S. Dokken, Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateElement.h"
#include <basix/finite-element.h>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/cell_types.h>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xview.hpp>

using namespace dolfinx;
using namespace dolfinx::fem;

namespace
{
// Computes the determinant of rectangular matrices
// det(A^T * A) = det(A) * det(A)
template <typename Matrix>
double compute_determinant(Matrix& A)
{
  if (A.shape(0) == A.shape(1))
    return math::det(A);
  else
  {
    auto ATA = xt::linalg::dot(xt::transpose(A), A);
    return std::sqrt(math::det(ATA));
  }
}
} // namespace

//-----------------------------------------------------------------------------
CoordinateElement::CoordinateElement(
    std::shared_ptr<basix::FiniteElement> element)
    : _element(element)
{
  int degree = _element->degree();
  const mesh::CellType cell = cell_shape();
  _is_affine = mesh::is_simplex(cell) and degree == 1;
}
//-----------------------------------------------------------------------------
CoordinateElement::CoordinateElement(mesh::CellType celltype, int degree)
    : CoordinateElement(
        std::make_shared<basix::FiniteElement>(basix::create_element(
            basix::element::family::P,
            basix::cell::str_to_type(mesh::to_string(celltype)), degree,
            basix::lattice::type::equispaced)))
{
  // Do nothing
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
xt::xtensor<double, 4>
CoordinateElement::tabulate(int n, const xt::xtensor<double, 2>& X) const
{
  return _element->tabulate(n, X);
}
//--------------------------------------------------------------------------------
void CoordinateElement::compute_jacobian(
    const xt::xtensor<double, 4>& dphi, const xt::xtensor<double, 2>& cell_geom,
    xt::xtensor<double, 3>& J) const
{
  // Number of points
  std::size_t num_points = dphi.shape(1);
  if (num_points == 0)
    return;

  // in-argument checks
  const std::size_t tdim = this->topological_dimension();
  const std::size_t gdim = cell_geom.shape(1);
  const std::size_t d = cell_geom.shape(0);

  // In/out size checks
  assert(J.shape(0) == num_points);
  assert(J.shape(1) == gdim);
  assert(J.shape(2) == tdim);
  assert(dphi.shape(0) == tdim);
  assert(dphi.shape(1) == num_points);
  assert(dphi.shape(3) == 1); // Assumes that value size is equal to 1

  xt::xtensor<double, 2> dphi0 = xt::empty<double>({tdim, d});
  if (_is_affine)
  {
    xt::noalias(dphi0) = xt::view(dphi, xt::all(), 0, xt::all(), 0);
    auto J0 = xt::linalg::dot(xt::transpose(cell_geom), xt::transpose(dphi0));
    J = xt::broadcast(J0, J.shape());
  }
  else
  {
    for (std::size_t p = 0; p < num_points; ++p)
    {
      xt::noalias(dphi0) = xt::view(dphi, xt::all(), p, xt::all(), 0);
      auto J_ip = xt::view(J, p, xt::all(), xt::all());
      auto J0 = xt::linalg::dot(xt::transpose(cell_geom), xt::transpose(dphi0));
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

  const int gdim = J.shape(1);
  const int tdim = K.shape(1);

  xt::xtensor<double, 2> K0 = xt::empty<double>({tdim, gdim});
  xt::xtensor<double, 2> J0 = xt::empty<double>({gdim, tdim});

  if (_is_affine)
  {
    J0 = xt::view(J, 0, xt::all(), xt::all());
    if (gdim == tdim)
      math::inv(J0, K0);
    else
      K0 = xt::linalg::pinv(J0);
    K = xt::broadcast(K0, K.shape());
  }
  else
  {
    for (std::size_t p = 0; p < J.shape(0); ++p)
    {
      J0 = xt::view(J, p, xt::all(), xt::all());
      if (gdim == tdim)
        math::inv(J0, K0);
      else
        K0 = xt::linalg::pinv(J0);
      auto K_ip = xt::view(K, p, xt::all(), xt::all());
      K_ip.assign(K0);
    }
  }
}
//--------------------------------------------------------------------------------
void CoordinateElement::compute_jacobian_determinant(
    const xt::xtensor<double, 3>& J, xt::xtensor<double, 1>& Jdet) const
{
  assert(J.shape(0) == Jdet.shape(0));
  if (_is_affine)
  {
    auto J0 = xt::view(J, 0, xt::all(), xt::all());
    double det = compute_determinant(J0);
    Jdet.fill(det);
  }
  else
  {
    for (std::size_t p = 0; p < J.shape(0); ++p)
    {
      auto Jp = xt::view(J, p, xt::all(), xt::all());
      double det = compute_determinant(Jp);
      Jdet[p] = det;
    }
  }
}
//-----------------------------------------------------------------------------
ElementDofLayout CoordinateElement::dof_layout() const
{
  assert(_element);
  std::vector<std::vector<std::set<int>>> entity_dofs = _element->entity_dofs();
  std::vector<std::vector<std::set<int>>> entity_closure_dofs
      = _element->entity_closure_dofs();

  return ElementDofLayout(1, entity_dofs, entity_closure_dofs, {}, {});
}
//-----------------------------------------------------------------------------
void CoordinateElement::push_forward(
    xt::xtensor<double, 2>& x, const xt::xtensor<double, 2>& cell_geometry,
    const xt::xtensor<double, 2>& phi)
{
  assert(phi.shape(1) == cell_geometry.shape(0));

  // Compute physical coordinates
  // x = phi * cell_geometry;
  x.fill(0);
  for (std::size_t i = 0; i < x.shape(0); ++i)
    for (std::size_t j = 0; j < x.shape(1); ++j)
      for (std::size_t k = 0; k < cell_geometry.shape(0); ++k)
        x(i, j) += phi(i, k) * cell_geometry(k, j);
}
//-----------------------------------------------------------------------------
void CoordinateElement::pull_back(
    xt::xtensor<double, 2>& X, xt::xtensor<double, 3>& J,
    xt::xtensor<double, 1>& detJ, xt::xtensor<double, 3>& K,
    const xt::xtensor<double, 2>& x,
    const xt::xtensor<double, 2>& cell_geometry) const
{
  // Number of points
  std::size_t num_points = x.shape(0);
  if (num_points == 0)
    return;

  // in-argument checks
  const std::size_t tdim = this->topological_dimension();
  const std::size_t gdim = x.shape(1);
  const std::size_t d = cell_geometry.shape(0);
  assert(cell_geometry.shape(1) == gdim);

  // In/out size checks
  assert(X.shape(0) == num_points);
  assert(X.shape(1) == tdim);
  assert(J.size() == num_points * gdim * tdim);
  assert(detJ.size() == num_points);
  assert(K.size() == num_points * gdim * tdim);

  xt::xtensor<double, 4> dphi({tdim, num_points, d, 1});
  if (_is_affine)
  {
    // Tabulate shape function and first derivative at the origin
    xt::xtensor<double, 2> X0 = xt::zeros<double>({std::size_t(1), tdim});
    xt::xtensor<double, 4> tabulated_data = _element->tabulate(1, X0);
    dphi = xt::view(tabulated_data, xt::range(1, tdim + 1), xt::all(),
                    xt::all(), xt::all());

    // Compute Jacobian, its inverse and determinant
    compute_jacobian(dphi, cell_geometry, J);
    compute_jacobian_inverse(J, K);
    compute_jacobian_determinant(J, detJ);

    // Compute physical coordinates at X=0 (phi(X) * cell_geom).
    auto phi0 = xt::view(tabulated_data, 0, 0, xt::all(), 0);
    auto x0 = xt::linalg::dot(xt::transpose(cell_geometry), phi0);

    // Calculate X for each point
    auto K0 = xt::view(K, 0, xt::all(), xt::all());
    for (std::size_t ip = 0; ip < num_points; ++ip)
      xt::row(X, ip) = xt::linalg::dot(K0, xt::row(x, ip) - x0);
  }
  else
  {
    xt::xtensor<double, 2> Xk({1, tdim});
    xt::xtensor<double, 1> dX = xt::empty<double>({tdim});
    for (std::size_t ip = 0; ip < num_points; ++ip)
    {
      Xk.fill(0);
      int k;
      for (k = 0; k < non_affine_max_its; ++k)
      {
        xt::xtensor<double, 4> tabulated_data = _element->tabulate(1, Xk);
        dphi = xt::view(tabulated_data, xt::range(1, tdim + 1), xt::all(),
                        xt::all(), xt::all());

        // cell_geometry * phi(0)
        auto phi0 = xt::view(tabulated_data, 0, 0, xt::all(), 0);
        auto xk = xt::linalg::dot(xt::transpose(cell_geometry), phi0);

        // Compute Jacobian, its inverse and determinant
        compute_jacobian(dphi, cell_geometry, J);
        compute_jacobian_inverse(J, K);
        compute_jacobian_determinant(J, detJ);

        auto K0 = xt::view(K, 0, xt::all(), xt::all());
        dX = xt::linalg::dot(K0, xt::row(x, ip) - xk);

        if (xt::linalg::norm(dX) < non_affine_atol)
          break;
        Xk += dX;
      }
      xt::row(X, ip) = xt::row(Xk, 0);
      if (k == non_affine_max_its)
      {
        throw std::runtime_error(
            "Newton method failed to converge for non-affine geometry");
      }
    }
  }
}
//-----------------------------------------------------------------------------
void CoordinateElement::permute_dofs(xtl::span<std::int32_t> dofs,
                                     const std::uint32_t cell_perm) const
{
  assert(_element);
  _element->permute_dofs(dofs, cell_perm);
}
//-----------------------------------------------------------------------------
void CoordinateElement::unpermute_dofs(xtl::span<std::int32_t> dofs,
                                       const std::uint32_t cell_perm) const
{
  assert(_element);
  _element->unpermute_dofs(dofs, cell_perm);
}
//-----------------------------------------------------------------------------
bool CoordinateElement::needs_dof_permutations() const
{
  assert(_element);
  assert(_element->dof_transformations_are_permutations());
  return !_element->dof_transformations_are_identity();
}
//-----------------------------------------------------------------------------
