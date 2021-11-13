// Copyright (C) 2020-2021 Garth N. Wells, Igor A. Baratta
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/fem/CoordinateElement.h>
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <functional>
#include <numeric>
#include <vector>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::mesh
{
class Mesh;
} // namespace dolfinx::mesh

namespace dolfinx::fem
{

template <typename T>
class Function;

/// Compute the evaluation points in the physical space at which an
/// expression should be computed to interpolate it in a finite elemenet
/// space.
///
/// @param[in] element The element to be interpolated into
/// @param[in] mesh The domain
/// @param[in] cells Indices of the cells in the mesh to compute
/// interpolation coordinates for
/// @return The coordinates in the physical space at which to evaluate
/// an expression
xt::xtensor<double, 2>
interpolation_coords(const fem::FiniteElement& element, const mesh::Mesh& mesh,
                     const xtl::span<const std::int32_t>& cells);

/// Interpolate an expression in a finite element space
///
/// @param[out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
/// @param[in] x The points at which f should be evaluated, as computed
/// by fem::interpolation_coords. The element used in
/// fem::interpolation_coords should be the same element as associated
/// with u.
/// @param[in] cells Indices of the cells in the mesh on which to
/// interpolate. Should be the same as the list used when calling
/// fem::interpolation_coords.
template <typename T>
void interpolate(
    Function<T>& u,
    const std::function<xt::xarray<T>(const xt::xtensor<double, 2>&)>& f,
    const xt::xtensor<double, 2>& x, const xtl::span<const std::int32_t>& cells)
{
  const std::shared_ptr<const FiniteElement> element
      = u.function_space()->element();
  assert(element);
  const int element_bs = element->block_size();
  if (int num_sub = element->num_sub_elements();
      num_sub > 0 and num_sub != element_bs)
  {
    throw std::runtime_error("Cannot directly interpolate a mixed space. "
                             "Interpolate into subspaces.");
  }

  // Get mesh
  assert(u.function_space());
  auto mesh = u.function_space()->mesh();
  assert(mesh);

  const int gdim = mesh->geometry().dim();
  const int tdim = mesh->topology().dim();

  // Get the interpolation points on the reference cells
  const xt::xtensor<double, 2>& X = element->interpolation_points();

  if (X.shape(0) == 0)
  {
    throw std::runtime_error(
        "Interpolation into this space is not yet supported.");
  }

  xtl::span<const std::uint32_t> cell_info;
  if (element->needs_dof_transformations())
  {
    mesh->topology_mutable().create_entity_permutations();
    cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
  }

  // Evaluate function at physical points. The returned array has a
  // number of rows equal to the number of components of the function,
  // and the number of columns is equal to the number of evaluation
  // points.
  xt::xarray<T> values = f(x);

  if (values.dimension() == 1)
  {
    if (element->value_size() != 1)
      throw std::runtime_error("Interpolation data has the wrong shape.");
    values.reshape(
        {static_cast<std::size_t>(element->value_size()), x.shape(1)});
  }

  if (values.shape(0) != element->value_size())
    throw std::runtime_error("Interpolation data has the wrong shape.");

  if (values.shape(1) != cells.size() * X.shape(0))
    throw std::runtime_error("Interpolation data has the wrong shape.");

  // Get dofmap
  const auto dofmap = u.function_space()->dofmap();
  assert(dofmap);
  const int dofmap_bs = dofmap->bs();

  // Loop over cells and compute interpolation dofs
  const int num_scalar_dofs = element->space_dimension() / element_bs;
  const int value_size = element->value_size() / element_bs;

  xtl::span<T> coeffs = u.x()->mutable_array();
  std::vector<T> _coeffs(num_scalar_dofs);

  const std::function<void(const xtl::span<T>&,
                           const xtl::span<const std::uint32_t>&, std::int32_t,
                           int)>
      apply_inverse_transpose_dof_transformation
      = element->get_dof_transformation_function<T>(true, true, true);

  // This assumes that any element with an identity interpolation matrix is a
  // point evaluation
  if (element->interpolation_ident())
  {
    for (std::int32_t c : cells)
    {
      xtl::span<const std::int32_t> dofs = dofmap->cell_dofs(c);
      for (int k = 0; k < element_bs; ++k)
      {
        for (int i = 0; i < num_scalar_dofs; ++i)
          _coeffs[i] = values(k, c * num_scalar_dofs + i);
        apply_inverse_transpose_dof_transformation(_coeffs, cell_info, c, 1);
        for (int i = 0; i < num_scalar_dofs; ++i)
        {
          const int dof = i * element_bs + k;
          std::div_t pos = std::div(dof, dofmap_bs);
          coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
        }
      }
    }
  }
  else
  {
    // Get coordinate map
    const fem::CoordinateElement& cmap = mesh->geometry().cmap();

    // Get geometry data
    const graph::AdjacencyList<std::int32_t>& x_dofmap
        = mesh->geometry().dofmap();
    // FIXME: Add proper interface for num coordinate dofs
    const int num_dofs_g = x_dofmap.num_links(0);
    const xt::xtensor<double, 2>& x_g = mesh->geometry().x();

    // Create data structures for Jacobian info
    xt::xtensor<double, 3> J = xt::empty<double>({int(X.shape(0)), gdim, tdim});
    xt::xtensor<double, 3> K = xt::empty<double>({int(X.shape(0)), tdim, gdim});
    xt::xtensor<double, 1> detJ = xt::empty<double>({X.shape(0)});

    xt::xtensor<double, 2> coordinate_dofs
        = xt::empty<double>({num_dofs_g, gdim});

    xt::xtensor<T, 3> reference_data({X.shape(0), 1, value_size});
    xt::xtensor<T, 3> _vals({X.shape(0), 1, value_size});

    // Tabulate 1st order derivatives of shape functions at interpolation coords
    xt::xtensor<double, 3> dphi = xt::view(
        cmap.tabulate(1, X), xt::range(1, tdim + 1), xt::all(), xt::all(), 0);

    const std::function<void(const xtl::span<T>&,
                             const xtl::span<const std::uint32_t>&,
                             std::int32_t, int)>
        apply_inverse_transpose_dof_transformation
        = element->get_dof_transformation_function<T>(true, true);

    for (std::int32_t c : cells)
    {
      auto x_dofs = x_dofmap.links(c);
      for (int i = 0; i < num_dofs_g; ++i)
        for (int j = 0; j < gdim; ++j)
          coordinate_dofs(i, j) = x_g(x_dofs[i], j);

      // Compute J, detJ and K
      J.fill(0);
      for (std::size_t p = 0; p < X.shape(0); ++p)
      {
        cmap.compute_jacobian(xt::view(dphi, xt::all(), p, xt::all()),
                              coordinate_dofs,
                              xt::view(J, p, xt::all(), xt::all()));
        cmap.compute_jacobian_inverse(xt::view(J, p, xt::all(), xt::all()),
                                      xt::view(K, p, xt::all(), xt::all()));
        detJ[p] = cmap.compute_jacobian_determinant(
            xt::view(J, p, xt::all(), xt::all()));
      }

      xtl::span<const std::int32_t> dofs = dofmap->cell_dofs(c);
      for (int k = 0; k < element_bs; ++k)
      {
        // Extract computed expression values for element block k
        for (int m = 0; m < value_size; ++m)
        {
          std::copy_n(&values(k * value_size + m, c * X.shape(0)), X.shape(0),
                      xt::view(_vals, xt::all(), 0, m).begin());
        }

        // Get element degrees of freedom for block
        element->map_pull_back(_vals, J, detJ, K, reference_data);

        xt::xtensor<T, 2> ref_data
            = xt::transpose(xt::view(reference_data, xt::all(), 0, xt::all()));
        element->interpolate(ref_data, tcb::make_span(_coeffs));
        apply_inverse_transpose_dof_transformation(_coeffs, cell_info, c, 1);

        assert(_coeffs.size() == num_scalar_dofs);

        // Copy interpolation dofs into coefficient vector
        for (int i = 0; i < num_scalar_dofs; ++i)
        {
          const int dof = i * element_bs + k;
          std::div_t pos = std::div(dof, dofmap_bs);
          coeffs[dofmap_bs * dofs[pos.quot] + pos.rem] = _coeffs[i];
        }
      }
    }
  }
}

/// Interpolate an expression f(x)
///
/// @note  This interface uses an expression function f that has an
/// in/out argument for the expression values. It is primarily to
/// support C code implementations of the expression, e.g. using Numba.
/// Generally the interface where the expression function is a pure
/// function, i.e. the expression values are the return argument, should
/// be preferred.
///
/// @param[out] u The function to interpolate into
/// @param[in] f The expression to be interpolated
/// @param[in] x The points at which should be evaluated, as
/// computed by fem::interpolation_coords
/// @param[in] cells Indices of the cells in the mesh on which to
/// interpolate. Should be the same as the list used when calling
/// fem::interpolation_coords.
template <typename T>
void interpolate_c(
    Function<T>& u,
    const std::function<void(xt::xarray<T>&, const xt::xtensor<double, 2>&)>& f,
    const xt::xtensor<double, 2>& x, const xtl::span<const std::int32_t>& cells)
{
  const std::shared_ptr<const FiniteElement> element
      = u.function_space()->element();
  assert(element);
  std::vector<int> vshape(element->value_rank(), 1);
  for (std::size_t i = 0; i < vshape.size(); ++i)
    vshape[i] = element->value_dimension(i);
  const std::size_t value_size = std::reduce(
      std::begin(vshape), std::end(vshape), 1, std::multiplies<>());

  auto fn = [value_size, &f](const xt::xtensor<double, 2>& x)
  {
    xt::xarray<T> values = xt::empty<T>({value_size, x.shape(1)});
    f(values, x);
    return values;
  };

  interpolate<T>(u, fn, x, cells);
}

/// Interpolate from one finite element Function to another on the same
/// mesh
/// @param[out] u The function to interpolate into
/// @param[in] v The function to be interpolated
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v)
{
  assert(v.function_space());
  const auto mesh = u.function_space()->mesh();
  assert(mesh);
  assert(v.function_space()->mesh());
  if (mesh != v.function_space()->mesh())
  {
    throw std::runtime_error(
        "Interpolation on different meshes not supported (yet).");
  }

  const int tdim = mesh->topology().dim();
  const std::shared_ptr<const FiniteElement> element_to
      = u.function_space()->element();
  assert(element_to);
  const std::shared_ptr<const FiniteElement> element_from
      = v.function_space()->element();
  assert(element_from);

  auto map = mesh->topology().index_map(tdim);
  assert(map);
  xtl::span<T> u_array = u.x()->mutable_array();
  xtl::span<const T> v_array = v.x()->array();
  if (u.function_space() == v.function_space())
  {
    // --- Same function spaces
    std::copy(v_array.begin(), v_array.end(), u_array.begin());
  }
  else if (element_to->hash() == element_from->hash())
  {
    // --- Same element

    // Get dofmaps
    assert(v.function_space());
    std::shared_ptr<const fem::DofMap> dofmap_v = v.function_space()->dofmap();
    assert(dofmap_v);
    const auto dofmap_u = u.function_space()->dofmap();
    assert(dofmap_u);

    // Iterate over mesh and interpolate on each cell
    const int num_cells = map->size_local() + map->num_ghosts();
    const int bs = dofmap_v->bs();
    assert(bs == dofmap_u->bs());
    for (int c = 0; c < num_cells; ++c)
    {
      xtl::span<const std::int32_t> dofs_v = dofmap_v->cell_dofs(c);
      xtl::span<const std::int32_t> dofs_u = dofmap_u->cell_dofs(c);
      assert(dofs_v.size() == dofs_u.size());
      for (std::size_t i = 0; i < dofs_v.size(); ++i)
        for (int k = 0; k < bs; ++k)
          u_array[bs * dofs_u[i] + k] = v_array[bs * dofs_v[i] + k];
    }
  }
  else
  {
    // --- Different elements

    xtl::span<const std::uint32_t> cell_info;
    if (element_to->needs_dof_transformations()
        or element_from->needs_dof_transformations())
    {
      mesh->topology_mutable().create_entity_permutations();
      cell_info = xtl::span(mesh->topology().get_cell_permutation_info());
    }

    // Get dofmaps
    const auto dofmap_u = u.function_space()->dofmap();
    const auto dofmap_v = v.function_space()->dofmap();

    // Create interpolation operator
    const xt::xtensor<double, 2> i_m
        = element_to->create_interpolation_operator(*element_from);

    // Get block sizes and dof transformation operators
    const int u_bs = element_to->block_size();
    const int v_bs = element_from->block_size();
    const auto apply_dof_transformation
        = element_from->get_dof_transformation_function<T>(false, true, false);
    const auto apply_inverse_dof_transform
        = element_to->get_dof_transformation_function<T>(true, true, false);

    // Creat working array
    std::vector<T> v_local(element_from->space_dimension());
    std::vector<T> u_local(element_to->space_dimension());

    // Iterate over mesh and interpolate on each cell
    const int num_cells = map->size_local() + map->num_ghosts();
    for (int c = 0; c < num_cells; ++c)
    {
      xtl::span<const std::int32_t> dofs_v = dofmap_v->cell_dofs(c);
      for (std::size_t i = 0; i < dofs_v.size(); i++)
        for (int k = 0; k < v_bs; k++)
          v_local[v_bs * i + k] = v_array[v_bs * dofs_v[i] + k];

      apply_dof_transformation(v_local, cell_info, c, 1);

      // FIXME: Get compile-time ranges from Basix
      // Apply interpolation operator
      std::fill(u_local.begin(), u_local.end(), 0);
      for (std::size_t i = 0; i < i_m.shape(0); ++i)
        for (std::size_t j = 0; j < i_m.shape(1); ++j)
          u_local[i] += i_m(i, j) * v_local[j];

      apply_inverse_dof_transform(u_local, cell_info, c, 1);

      xtl::span<const std::int32_t> dofs_u = dofmap_u->cell_dofs(c);
      for (std::size_t i = 0; i < dofs_u.size(); ++i)
        for (int k = 0; k < u_bs; ++k)
          u_array[u_bs * dofs_u[i] + k] = u_local[u_bs * i + k];
    }
  }
}

} // namespace dolfinx::fem
