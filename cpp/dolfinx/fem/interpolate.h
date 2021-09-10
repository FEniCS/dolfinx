// Copyright (C) 2020-2021 Garth N. Wells, Massimiliano Leoni
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "FunctionSpace.h"
#include <dolfinx/fem/DofMap.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/geometry/BoundingBoxTree.h>
#include <dolfinx/geometry/utils.h>
#include <dolfinx/mesh/Mesh.h>
#include <functional>
#include <numeric>
#include <variant>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

#include <dolfinx/common/log.h>

namespace dolfinx::fem
{

template <typename T>
class Function;

/// This should be hidden somewhere
template <typename T>
const MPI_Datatype MPI_TYPE = MPI_DOUBLE;

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

/// Interpolate a finite element Function (on possibly non-matching
/// meshes) in another finite element space
/// @param[out] u The function to interpolate into
/// @param[in] v The function to be interpolated
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v);

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
    const xt::xtensor<double, 2>& x,
    const xtl::span<const std::int32_t>& cells);

/// Interpolate function values in a finite element space
///
/// @param[out] u The function to interpolate into
/// @param[in] values The values of the interpolated function at points @p x
/// @param[in] x The points at which the interpolated function was evaluated,
/// as computed by fem::interpolation_coords. The element used in
/// fem::interpolation_coords should be the same element as associated
/// with u.
/// @param[in] cells Indices of the cells in the mesh on which to
/// interpolate. Should be the same as the list used when calling
/// fem::interpolation_coords.
template <typename T>
void interpolate(Function<T>& u, xt::xarray<T>& values,
                 const xt::xtensor<double, 2>& x,
                 const xtl::span<const std::int32_t>& cells);

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
    const xt::xtensor<double, 2>& x,
    const xtl::span<const std::int32_t>& cells);

namespace detail
{

template <typename T>
void interpolate_from_any(Function<T>& u, const Function<T>& v)
{
  assert(v.function_space());
  const auto element = u.function_space()->element();
  assert(element);
  if (v.function_space()->element()->hash() != element->hash())
  {
    throw std::runtime_error("Restricting finite elements function in "
                             "different elements not supported.");
  }

  const auto mesh = u.function_space()->mesh();
  assert(mesh);
  assert(v.function_space()->mesh());
  if (mesh->id() != v.function_space()->mesh()->id())
  {
    const int nProcs = dolfinx::MPI::size(MPI_COMM_WORLD);
    const auto mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

    const int tdim = u.function_space()->mesh()->topology().dim();
    const auto cell_map
        = u.function_space()->mesh()->topology().index_map(tdim);
    const int num_cells = cell_map->size_local() + cell_map->num_ghosts();
    std::vector<std::int32_t> cells(num_cells, 0);
    std::iota(cells.begin(), cells.end(), 0);

    // Collect all the points at which values are needed to define the
    // interpolating function
    const xt::xtensor<double, 2> x = fem::interpolation_coords(
        *u.function_space()->element(), *u.function_space()->mesh(), cells);

    // This transposition is a quick and dirty solution and should be avoided
    auto x_t = xt::zeros_like(xt::transpose(x));
    for (decltype(x.shape(1)) i = 0; i < x.shape(1); ++i)
    {
      for (decltype(x.shape(0)) j = 0; j < x.shape(0); ++j)
      {
        x_t(i, j) = x(j, i);
      }
    }

    dolfinx::geometry::BoundingBoxTree bb(*v.function_space()->mesh(), tdim,
                                          0.0001);
    auto globalBB = bb.create_global_tree(MPI_COMM_WORLD);

    std::vector<std::vector<int>> candidates(x_t.shape(0));
    for (decltype(x_t.shape(0)) i = 0; i < x_t.shape(0); ++i)
    {
      const auto xp = x_t(i, 0);
      const auto yp = x_t(i, 1);
      const auto zp = x_t(i, 2);
      candidates[i]
          = dolfinx::geometry::compute_collisions(globalBB, {xp, yp, zp});
    }

    std::vector<std::vector<double>> pointsToSend(nProcs);
    std::for_each(pointsToSend.begin(), pointsToSend.end(),
                  [&candidates](auto& el)
                  { el.reserve(3 * candidates.size()); });
    for (decltype(candidates.size()) i = 0; i < candidates.size(); ++i)
    {
      for (const auto& p : candidates[i])
      {
        auto point = xt::row(x_t, i);
        pointsToSend[p].insert(pointsToSend[p].end(), point.begin(),
                               point.end());
      }
    }

    std::vector<std::int32_t> nPointsToSend(nProcs);
    std::transform(pointsToSend.cbegin(), pointsToSend.cend(),
                   nPointsToSend.begin(),
                   [](const auto& el) { return el.size(); });

    std::vector<std::int32_t> allPointsToSend(nProcs * nProcs);
    MPI_Allgather(nPointsToSend.data(), nProcs, MPI_INT32_T,
                  allPointsToSend.data(), nProcs, MPI_INT32_T, MPI_COMM_WORLD);

    std::size_t nPointsToReceive = 0;
    for (int i = 0; i < nProcs; ++i)
    {
      nPointsToReceive += allPointsToSend[mpi_rank + i * nProcs];
    }
    std::vector<std::int32_t> sendingOffsets(nProcs, 0);
    for (int i = 0; i < nProcs; ++i)
    {
      for (int j = 0; j < mpi_rank; ++j)
      {
        sendingOffsets[i] += allPointsToSend[nProcs * j + i];
      }
    }

    xt::xtensor<double, 2> pointsToReceive(
        xt::shape(
            {nPointsToReceive / 3, static_cast<decltype(nPointsToReceive)>(3)}),
        0);
    MPI_Win window;
    MPI_Win_create(pointsToReceive.data(), sizeof(double) * nPointsToReceive,
                   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);

    for (int i = 0; i < nProcs; ++i)
    {
      MPI_Put(pointsToSend[i].data(), pointsToSend[i].size(), MPI_DOUBLE, i,
              sendingOffsets[i], pointsToSend[i].size(), MPI_DOUBLE, window);
    }

    MPI_Win_fence(0, window);
    MPI_Win_free(&window);

    // Each process will now check at which points it can evaluate
    // the interpolating function, and note that down in evaluationCells
    std::vector<std::int32_t> evaluationCells(pointsToReceive.shape(0), -1);

    const auto connectivity = v.function_space()->mesh()->geometry().dofmap();

    // This BBT is useful for fast lookup of which cell contains a given point
    dolfinx::geometry::BoundingBoxTree bbt(*v.function_space()->mesh(), tdim,
                                           0.0001);

    const auto xv = v.function_space()->mesh()->geometry().x();

    // For each point at which the source function needs to be evaluated
    for (decltype(pointsToReceive.shape(0)) i = 0; i < pointsToReceive.shape(0);
         ++i)
    {
      // Get its coordinates
      const double xp = pointsToReceive(i, 0);
      const double yp = pointsToReceive(i, 1);
      const double zp = pointsToReceive(i, 2);

      const auto intersectingCells = dolfinx::geometry::select_colliding_cells(
          *v.function_space()->mesh(),
          dolfinx::geometry::compute_collisions(bbt, {xp, yp, zp}),
          {xp, yp, zp}, 1);

      if (not intersectingCells.empty())
      {
        evaluationCells[i] = intersectingCells[0];
      }
    }

    const auto value_size = u.function_space()->element()->value_size();

    xt::xtensor<T, 2> values(
        xt::shape(
            {pointsToReceive.shape(0),
             static_cast<decltype(pointsToReceive.shape(0))>(value_size)}),
        0);
    v.eval(pointsToReceive, evaluationCells, values);

    xt::xtensor<T, 2> valuesToRetrieve(
        xt::shape({std::accumulate(nPointsToSend.cbegin(), nPointsToSend.cend(),
                                   static_cast<size_t>(0))
                       / 3,
                   static_cast<size_t>(value_size)}),
        0);

    std::vector<size_t> retrievingOffsets(nProcs, 0);
    std::partial_sum(nPointsToSend.cbegin(), std::prev(nPointsToSend.cend()),
                     std::next(retrievingOffsets.begin()));
    std::transform(retrievingOffsets.cbegin(), retrievingOffsets.cend(),
                   retrievingOffsets.begin(),
                   [&value_size](const auto& el)
                   { return el / 3 * value_size; });

    MPI_Win_create(values.data(), sizeof(MPI_TYPE<T>) * values.size(),
                   sizeof(MPI_TYPE<T>), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);

    for (int i = 0; i < nProcs; ++i)
    {
      MPI_Get(valuesToRetrieve.data() + retrievingOffsets[i],
              pointsToSend[i].size() / 3 * value_size, MPI_TYPE<T>, i,
              sendingOffsets[i] / 3 * value_size,
              pointsToSend[i].size() / 3 * value_size, MPI_TYPE<T>, window);
    }

    MPI_Win_fence(0, window);
    MPI_Win_free(&window);

    xt::xarray<T> myVals(
        {x_t.shape(0), static_cast<decltype(x_t.shape(0))>(value_size)},
        static_cast<T>(0));

    std::vector<size_t> scanningProgress(nProcs, 0);

    for (decltype(candidates.size()) i = 0; i < candidates.size(); ++i)
    {
      for (const auto& p : candidates[i])
      {
        for (int j = 0; j < value_size; ++j)
        {
          if (myVals(i, j) == static_cast<T>(0))
          {
            myVals(i, j) = valuesToRetrieve(
                retrievingOffsets[p] / value_size + scanningProgress[p], j);
          }
        }
        ++scanningProgress[p];
      }
    }

    // This transposition is a quick and dirty solution and should be avoided
    xt::xarray<T> myVals_t = xt::zeros_like(xt::transpose(myVals));
    for (decltype(myVals.shape(1)) i = 0; i < myVals.shape(1); ++i)
    {
      for (decltype(myVals.shape(0)) j = 0; j < myVals.shape(0); ++j)
      {
        myVals_t(i, j) = myVals(j, i);
      }
    }

    // Finally, interpolate using the computed values
    fem::interpolate(u, myVals_t, x, cells);

    return;
  }
  const int tdim = mesh->topology().dim();

  // Get dofmaps
  assert(v.function_space());
  std::shared_ptr<const fem::DofMap> dofmap_v = v.function_space()->dofmap();
  assert(dofmap_v);
  auto map = mesh->topology().index_map(tdim);
  assert(map);

  std::vector<T>& coeffs = u.x()->mutable_array();

  // Iterate over mesh and interpolate on each cell
  const auto dofmap_u = u.function_space()->dofmap();
  const std::vector<T>& v_array = v.x()->array();
  const int num_cells = map->size_local() + map->num_ghosts();
  const int bs = dofmap_v->bs();
  assert(bs == dofmap_u->bs());
  for (int c = 0; c < num_cells; ++c)
  {
    xtl::span<const std::int32_t> dofs_v = dofmap_v->cell_dofs(c);
    xtl::span<const std::int32_t> cell_dofs = dofmap_u->cell_dofs(c);
    assert(dofs_v.size() == cell_dofs.size());
    for (std::size_t i = 0; i < dofs_v.size(); ++i)
    {
      for (int k = 0; k < bs; ++k)
        coeffs[bs * cell_dofs[i] + k] = v_array[bs * dofs_v[i] + k];
    }
  }
}

} // namespace detail

//----------------------------------------------------------------------------
template <typename T>
void interpolate(Function<T>& u, const Function<T>& v)
{
  assert(u.function_space());
  const std::shared_ptr<const FiniteElement> element
      = u.function_space()->element();
  assert(element);

  // Check that function ranks match
  if (int rank_v = v.function_space()->element()->value_rank();
      element->value_rank() != rank_v)
  {
    throw std::runtime_error("Cannot interpolate function into function space. "
                             "Rank of function ("
                             + std::to_string(rank_v)
                             + ") does not match rank of function space ("
                             + std::to_string(element->value_rank()) + ")");
  }

  // Check that function dimension match
  for (int i = 0; i < element->value_rank(); ++i)
  {
    if (int v_dim = v.function_space()->element()->value_dimension(i);
        element->value_dimension(i) != v_dim)
    {
      throw std::runtime_error(
          "Cannot interpolate function into function space. "
          "Dimension "
          + std::to_string(i) + " of function (" + std::to_string(v_dim)
          + ") does not match dimension " + std::to_string(i)
          + " of function space (" + std::to_string(element->value_dimension(i))
          + ")");
    }
  }

  detail::interpolate_from_any(u, v);
}
//----------------------------------------------------------------------------
template <typename T>
void interpolate(Function<T>& u, xt::xarray<T>& values,
                 const xt::xtensor<double, 2>& x,
                 const xtl::span<const std::int32_t>& cells)
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

  if (values.dimension() == 1)
  {
    if (element->value_size() != 1)
      throw std::runtime_error("Interpolation data has the wrong shape.");
    values.reshape(
        {static_cast<std::size_t>(element->value_size()), x.shape(1)});
  }

  if (values.shape(0) != static_cast<std::size_t>(element->value_size()))
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

  std::vector<T>& coeffs = u.x()->mutable_array();
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
    xt::xtensor<double, 4> dphi
        = xt::view(cmap.tabulate(1, X), xt::range(1, tdim + 1), xt::all(),
                   xt::all(), xt::all());

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
      cmap.compute_jacobian(dphi, coordinate_dofs, J);
      cmap.compute_jacobian_inverse(J, K);
      cmap.compute_jacobian_determinant(J, detJ);

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
//----------------------------------------------------------------------------
template <typename T>
void interpolate(
    Function<T>& u,
    const std::function<xt::xarray<T>(const xt::xtensor<double, 2>&)>& f,
    const xt::xtensor<double, 2>& x, const xtl::span<const std::int32_t>& cells)
{
  // Evaluate function at physical points. The returned array has a
  // number of rows equal to the number of components of the function,
  // and the number of columns is equal to the number of evaluation
  // points.
  xt::xarray<T> values = f(x);
  interpolate(u, values, x, cells);
}
//----------------------------------------------------------------------------
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
//----------------------------------------------------------------------------

} // namespace dolfinx::fem
