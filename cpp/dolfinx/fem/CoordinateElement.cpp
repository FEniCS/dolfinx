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
CoordinateElement::CoordinateElement(mesh::CellType celltype, int degree,
                                     basix::element::lagrange_variant type)
    : CoordinateElement(std::make_shared<basix::FiniteElement>(
        basix::create_element(basix::element::family::P,
                              mesh::cell_type_to_basix_type(celltype), degree,
                              type, false)))
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
std::array<std::size_t, 4>
CoordinateElement::tabulate_shape(std::size_t nd, std::size_t num_points) const
{
  return _element->tabulate_shape(nd, num_points);
}
//-----------------------------------------------------------------------------
xt::xtensor<double, 4>
CoordinateElement::tabulate(int n, const xt::xtensor<double, 2>& X) const
{
  return _element->tabulate(n, X);
}
//--------------------------------------------------------------------------------
void CoordinateElement::tabulate(int n, const xt::xtensor<double, 2>& X,
                                 xt::xtensor<double, 4>& basis) const
{
  _element->tabulate(n, X, basis);
}
//--------------------------------------------------------------------------------
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
std::array<double, 3>
CoordinateElement::x0(const xt::xtensor<double, 2>& cell_geometry)
{
  std::array<double, 3> x0 = {0, 0, 0};
  for (std::size_t i = 0; i < cell_geometry.shape(1); ++i)
    x0[i] += cell_geometry(0, i);
  return x0;
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
  xt::xtensor<double, 4> basis(_element->tabulate_shape(1, 1));
  for (std::size_t p = 0; p < num_points; ++p)
  {
    Xk.fill(0);
    int k;
    for (k = 0; k < maxit; ++k)
    {
      _element->tabulate(1, Xk, basis);

      // x = cell_geometry * phi
      auto phi = xt::view(basis, 0, 0, xt::all(), 0);
      std::fill(xk.begin(), xk.end(), 0.0);
      for (std::size_t i = 0; i < cell_geometry.shape(1); ++i)
        for (std::size_t j = 0; j < cell_geometry.shape(0); ++j)
          xk[i] += cell_geometry(j, i) * phi[j];

      // Compute Jacobian, its inverse and determinant
      J.fill(0);
      dphi = xt::view(basis, xt::range(1, tdim + 1), 0, xt::all(), 0);
      compute_jacobian(dphi, cell_geometry, J);
      compute_jacobian_inverse(J, K);

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
