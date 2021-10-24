// Copyright (C) 2018-2021 Garth N. Wells, Jørgen S. Dokken, Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CoordinateElement.h"
#include <basix/finite-element.h>
#include <cmath>
#include <dolfinx/common/math.h>
#include <dolfinx/mesh/cell_types.h>
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
    using T = typename Matrix::value_type;
    xt::xtensor<T, 2> B = xt::transpose(A);
    xt::xtensor<T, 2> BA = xt::zeros<T>({B.shape(0), A.shape(1)});
    math::dot(B, A, BA);
    return std::sqrt(math::det(BA));
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
            basix::element::family::P, mesh::cell_type_to_basix_type(celltype),
            degree, basix::element::lagrange_variant::equispaced, false)))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
mesh::CellType CoordinateElement::cell_shape() const
{
  return mesh::cell_type_from_basix_type(_element->cell_type());
}
//-----------------------------------------------------------------------------
int CoordinateElement::topological_dimension() const
{
  return basix::cell::topological_dimension(_element->cell_type());
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
  xt::xtensor<double, 2> J0 = xt::zeros<double>({gdim, tdim});
  xt::xtensor<double, 2> dphi0 = xt::empty<double>({tdim, d});
  if (_is_affine)
  {
    xt::noalias(dphi0) = xt::view(dphi, xt::all(), 0, xt::all(), 0);
    math::dot(cell_geom, dphi0, J0, true);
    // NOTE: Should be using xt::broadcast, but it's much slower than a
    // plain loop.
    for (std::size_t p = 0; p < num_points; ++p)
    {
      auto J_ip = xt::view(J, p, xt::all(), xt::all());
      J_ip.assign(J0);
    }
  }
  else
  {
    for (std::size_t p = 0; p < num_points; ++p)
    {
      J0.fill(0);
      xt::noalias(dphi0) = xt::view(dphi, xt::all(), p, xt::all(), 0);
      auto J_ip = xt::view(J, p, xt::all(), xt::all());
      math::dot(cell_geom, dphi0, J0, true);
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
  xt::xtensor<double, 2> K0 = xt::zeros<double>({tdim, gdim});
  xt::xtensor<double, 2> J0 = xt::zeros<double>({gdim, tdim});
  if (_is_affine)
  {
    J0 = xt::view(J, 0, xt::all(), xt::all());
    if (gdim == tdim)
      math::inv(J0, K0);
    else
      math::pinv(J0, K0);
    for (std::size_t p = 0; p < J.shape(0); ++p)
    {
      auto K_ip = xt::view(K, p, xt::all(), xt::all());
      K_ip.assign(K0);
    }
  }
  else
  {
    for (std::size_t p = 0; p < J.shape(0); ++p)
    {
      K0.fill(0);
      J0 = xt::view(J, p, xt::all(), xt::all());
      if (gdim == tdim)
        math::inv(J0, K0);
      else
        math::pinv(J0, K0);
      auto K_ip = xt::view(K, p, xt::all(), xt::all());
      K_ip.assign(K0);
    }
  }
}
//--------------------------------------------------------------------------------
void CoordinateElement::compute_jacobian_inverse(
    const xt::xtensor<double, 2>& J, xt::xtensor<double, 2>& K) const
{
  const int gdim = J.shape(1);
  const int tdim = K.shape(1);
  if (gdim == tdim)
    math::inv(J, K);
  else
    math::pinv(J, K);
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
  std::vector<std::vector<std::vector<int>>> entity_dofs
      = _element->entity_dofs();
  std::vector<std::vector<std::vector<int>>> entity_closure_dofs
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
  math::dot(phi, cell_geometry, x);
}
//-----------------------------------------------------------------------------
void CoordinateElement::pull_back_affine(xt::xtensor<double, 2>& X,
                                         const xt::xtensor<double, 2>& K,
                                         const std::array<double, 3>& x0,
                                         const xt::xtensor<double, 2>& x)
{
  assert(X.shape(0) == x.shape(0));
  assert(X.shape(1) == K.shape(0));
  assert(x.shape(1) == K.shape(1));

  // Calculate X for each point
  X.fill(0.0);
  for (std::size_t p = 0; p < x.shape(0); ++p)
    for (std::size_t i = 0; i < K.shape(0); ++i)
      for (std::size_t j = 0; j < K.shape(1); ++j)
        X(p, i) += K(i, j) * (x(p, j) - x0[j]);
}
//-----------------------------------------------------------------------------
void CoordinateElement::pull_back_nonaffine(
    xt::xtensor<double, 2>& X, const xt::xtensor<double, 2>& x,
    const xt::xtensor<double, 2>& cell_geometry, double tol, int maxit) const
{
  // Number of points
  std::size_t num_points = x.shape(0);
  if (num_points == 0)
    return;

  const std::size_t tdim = this->topological_dimension();
  const std::size_t gdim = x.shape(1);
  const std::size_t num_xnodes = cell_geometry.shape(0);
  assert(cell_geometry.shape(1) == gdim);
  assert(X.shape(0) == num_points);
  assert(X.shape(1) == tdim);

  xt::xtensor<double, 2> dphi({tdim, num_xnodes});
  xt::xtensor<double, 2> Xk({1, tdim});
  std::array<double, 3> xk = {0, 0, 0};
  xt::xtensor<double, 1> dX = xt::empty<double>({tdim});
  xt::xtensor<double, 2> J({gdim, tdim});
  xt::xtensor<double, 2> K({tdim, gdim});
  for (std::size_t p = 0; p < num_points; ++p)
  {
    Xk.fill(0);
    int k;
    for (k = 0; k < maxit; ++k)
    {
      xt::xtensor<double, 4> basis = _element->tabulate(1, Xk);
      dphi = xt::view(basis, xt::range(1, tdim + 1), 0, xt::all(), 0);

      // x = cell_geometry * phi
      auto phi = xt::view(basis, 0, 0, xt::all(), 0);
      std::fill(xk.begin(), xk.end(), 0.0);
      for (std::size_t i = 0; i < cell_geometry.shape(1); ++i)
        for (std::size_t j = 0; j < cell_geometry.shape(0); ++j)
          xk[i] += cell_geometry(j, i) * phi[j];

      // Compute Jacobian, its inverse and determinant
      J.fill(0);
      compute_jacobian_new(dphi, cell_geometry, J);
      compute_jacobian_inverse_new(J, K);

      dX.fill(0.0);
      for (std::size_t i = 0; i < K.shape(0); ++i)
        for (std::size_t j = 0; j < K.shape(1); ++j)
          dX[i] += K(i, j) * (x(p, j) - xk[j]);

      Xk += dX;
      if (std::sqrt(xt::sum(dX * dX)()) < tol)
        break;
    }
    xt::row(X, p) = xt::row(Xk, 0);
    if (k == maxit)
    {
      throw std::runtime_error(
          "Newton method failed to converge for non-affine geometry");
    }
  }
}
//-----------------------------------------------------------------------------
void CoordinateElement::pull_back(
    xt::xtensor<double, 2>& X, const xt::xtensor<double, 2>& x,
    const xt::xtensor<double, 2>& cell_geometry) const
{
  // Number of points
  std::size_t num_points = x.shape(0);
  if (num_points == 0)
    return;

  // in-argument checks
  const std::size_t tdim = this->topological_dimension();
  const std::size_t gdim = x.shape(1);
  // const std::size_t d = cell_geometry.shape(0);
  assert(cell_geometry.shape(1) == gdim);

  // In/out size checks
  assert(X.shape(0) == num_points);
  assert(X.shape(1) == tdim);

  if (_is_affine)
  {
    // Tabulate shape function and first derivative at the origin
    xt::xtensor<double, 2> X0 = xt::zeros<double>({std::size_t(1), tdim});
    xt::xtensor<double, 4> data = _element->tabulate(1, X0);
    xt::xtensor<double, 2> dphi
        = xt::view(data, xt::range(1, tdim + 1), 0, xt::all(), 0);

    xt::xtensor<double, 2> J = xt::zeros<double>({gdim, tdim});
    xt::xtensor<double, 2> K = xt::zeros<double>({tdim, gdim});

    // Compute Jacobian, its inverse and determinant
    compute_jacobian_new(dphi, cell_geometry, J);
    compute_jacobian_inverse(J, K);

    // Compute physical coordinates at X=0 (phi(X) * cell_geom).
    auto phi0 = xt::view(data, 0, 0, xt::all(), 0);
    std::array<double, 3> x0 = {0, 0, 0};
    for (std::size_t i = 0; i < x.size(); ++i)
      for (std::size_t j = 0; j < phi0.shape(0); ++j)
        x0[i] += cell_geometry(j, i) * phi0[j];

    // Calculate X for each point
    pull_back_affine(X, K, x0, x);
  }
  else
  {
    pull_back_nonaffine(X, x, cell_geometry, non_affine_atol,
                        non_affine_max_its);
  }
}
//-----------------------------------------------------------------------------
void CoordinateElement::permute_dofs(const xtl::span<std::int32_t>& dofs,
                                     std::uint32_t cell_perm) const
{
  assert(_element);
  _element->permute_dofs(dofs, cell_perm);
}
//-----------------------------------------------------------------------------
void CoordinateElement::unpermute_dofs(const xtl::span<std::int32_t>& dofs,
                                       std::uint32_t cell_perm) const
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
