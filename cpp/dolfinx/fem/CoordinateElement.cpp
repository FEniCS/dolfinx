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

using namespace dolfinx;
using namespace dolfinx::fem;

//-----------------------------------------------------------------------------
template <std::floating_point T>
CoordinateElement<T>::CoordinateElement(
    std::shared_ptr<const basix::FiniteElement<T>> element)
    : _element(element)
{
  int degree = _element->degree();
  mesh::CellType cell = this->cell_shape();
  _is_affine = mesh::is_simplex(cell) and degree == 1;
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
CoordinateElement<T>::CoordinateElement(mesh::CellType celltype, int degree,
                                        basix::element::lagrange_variant type)
    : CoordinateElement(
        std::make_shared<basix::FiniteElement<T>>(basix::create_element<T>(
            basix::element::family::P, mesh::cell_type_to_basix_type(celltype),
            degree, type, basix::element::dpc_variant::unset, false)))
{
  // Do nothing
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
mesh::CellType CoordinateElement<T>::cell_shape() const
{
  return mesh::cell_type_from_basix_type(_element->cell_type());
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
std::array<std::size_t, 4>
CoordinateElement<T>::tabulate_shape(std::size_t nd,
                                     std::size_t num_points) const
{
  return _element->tabulate_shape(nd, num_points);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
void CoordinateElement<T>::tabulate(int nd, std::span<const T> X,
                                    std::array<std::size_t, 2> shape,
                                    std::span<T> basis) const
{
  assert(_element);
  _element->tabulate(nd, X, shape, basis);
}
//--------------------------------------------------------------------------------
template <std::floating_point T>
ElementDofLayout CoordinateElement<T>::create_dof_layout() const
{
  assert(_element);
  return ElementDofLayout(1, _element->entity_dofs(),
                          _element->entity_closure_dofs(), {}, {});
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
void CoordinateElement<T>::pull_back_nonaffine(mdspan2_t<T> X,
                                               mdspan2_t<const T> x,
                                               mdspan2_t<const T> cell_geometry,
                                               double tol, int maxit) const
{
  // Number of points
  std::size_t num_points = x.extent(0);
  if (num_points == 0)
    return;

  const std::size_t tdim = mesh::cell_dim(this->cell_shape());
  const std::size_t gdim = x.extent(1);
  const std::size_t num_xnodes = cell_geometry.extent(0);
  assert(cell_geometry.extent(1) == gdim);
  assert(X.extent(0) == num_points);
  assert(X.extent(1) == tdim);

  std::vector<T> dphi_b(tdim * num_xnodes);
  mdspan2_t<T> dphi(dphi_b.data(), tdim, num_xnodes);

  std::vector<T> Xk_b(tdim);
  mdspan2_t<T> Xk(Xk_b.data(), 1, tdim);

  std::array<T, 3> xk = {0, 0, 0};
  std::vector<T> dX(tdim);
  std::vector<T> J_b(gdim * tdim);
  mdspan2_t<T> J(J_b.data(), gdim, tdim);
  std::vector<T> K_b(tdim * gdim);
  mdspan2_t<T> K(K_b.data(), tdim, gdim);

  using mdspan4_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
      T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, 4>>;

  const std::array<std::size_t, 4> bsize = _element->tabulate_shape(1, 1);
  std::vector<T> basis_b(
      std::reduce(bsize.begin(), bsize.end(), 1, std::multiplies{}));
  mdspan4_t basis(basis_b.data(), bsize);
  std::vector<T> phi(basis.extent(2));
  for (std::size_t p = 0; p < num_points; ++p)
  {
    std::fill(Xk_b.begin(), Xk_b.end(), 0.0);
    int k;
    for (k = 0; k < maxit; ++k)
    {
      _element->tabulate(1, Xk_b, {1, tdim}, basis_b);

      // x = cell_geometry * phi
      std::fill(xk.begin(), xk.end(), 0.0);
      for (std::size_t i = 0; i < cell_geometry.extent(0); ++i)
        for (std::size_t j = 0; j < cell_geometry.extent(1); ++j)
          xk[j] += cell_geometry(i, j) * basis(0, 0, i, 0);

      // Compute Jacobian, its inverse and determinant
      std::fill(J_b.begin(), J_b.end(), 0.0);
      for (std::size_t i = 0; i < tdim; ++i)
        for (std::size_t j = 0; j < basis.extent(2); ++j)
          dphi(i, j) = basis(i + 1, 0, j, 0);

      compute_jacobian(dphi, cell_geometry, J);
      compute_jacobian_inverse(J, K);

      // Compute dX = K * (x_p - x_k)
      std::fill(dX.begin(), dX.end(), 0);
      for (std::size_t i = 0; i < K.extent(0); ++i)
        for (std::size_t j = 0; j < K.extent(1); ++j)
          dX[i] += K(i, j) * (x(p, j) - xk[j]);

      // Compute Xk += dX
      std::transform(dX.begin(), dX.end(), Xk_b.begin(), Xk_b.begin(),
                     [](auto a, auto b) { return a + b; });

      // Compute norm(dX)
      if (auto dX_squared
          = std::transform_reduce(dX.cbegin(), dX.cend(), 0.0, std::plus{},
                                  [](auto v) { return v * v; });
          std::sqrt(dX_squared) < tol)
      {
        break;
      }
    }

    std::copy(Xk_b.cbegin(), std::next(Xk_b.cbegin(), tdim),
              X.data_handle() + p * tdim);
    if (k == maxit)
    {
      throw std::runtime_error(
          "Newton method failed to converge for non-affine geometry");
    }
  }
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
void CoordinateElement<T>::permute_dofs(std::span<std::int32_t> dofs,
                                        std::uint32_t cell_perm) const
{
  assert(_element);
  _element->permute_dofs(dofs, cell_perm);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
void CoordinateElement<T>::unpermute_dofs(std::span<std::int32_t> dofs,
                                          std::uint32_t cell_perm) const
{
  assert(_element);
  _element->unpermute_dofs(dofs, cell_perm);
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
bool CoordinateElement<T>::needs_dof_permutations() const
{
  assert(_element);
  assert(_element->dof_transformations_are_permutations());
  return !_element->dof_transformations_are_identity();
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
int CoordinateElement<T>::degree() const
{
  assert(_element);
  return _element->degree();
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
int CoordinateElement<T>::dim() const
{
  assert(_element);
  return _element->dim();
}
//-----------------------------------------------------------------------------
template <std::floating_point T>
basix::element::lagrange_variant CoordinateElement<T>::variant() const
{
  assert(_element);
  return _element->lagrange_variant();
}
//-----------------------------------------------------------------------------
template class fem::CoordinateElement<float>;
template class fem::CoordinateElement<double>;
//-----------------------------------------------------------------------------
