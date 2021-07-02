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
#include <variant>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>
#include <xtl/xspan.hpp>

namespace dolfinx::fem
{

template <typename T>
class Function;

/// This should be hidden somewhere
template <typename T>
const MPI_Datatype MPI_TYPE = MPI_DOUBLE;

/// This function is used to define a custom MPI reduction operator and
/// it conforms to the MPI standard's specifications,
/// see https://www.mpi-forum.org/docs/mpi-3.1/mpi31-report/node115.htm
///
/// This function gathers values from different processors ignoring duplicates
///
/// This function should probably go into a detail namespace
///
/// @param[in] invec A vector of operands
/// @param[out] inoutvec A vector of operands, also used to store
/// the operation's result
/// @param[in] len The length of the first two arrays
/// @param[in] dt Not currently used
void SINGLESUM(void* invec, void* inoutvec, int* len, MPI_Datatype* dt);

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
    auto mpi_rank = dolfinx::MPI::rank(MPI_COMM_WORLD);

    const int tdim = u.function_space()->mesh()->topology().dim();
    auto cell_map = u.function_space()->mesh()->topology().index_map(tdim);
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

    // Gather a vector nProcs whose element i contains the number
    // of points needed by process i
    int nProcs = dolfinx::MPI::size(MPI_COMM_WORLD);
    std::vector<int> nPoints(nProcs, -1);
    nPoints[mpi_rank] = x.shape(1);

    MPI_Allgather(nPoints.data() + mpi_rank, 1, MPI_INT, nPoints.data(), 1,
                  MPI_INT, MPI_COMM_WORLD);

    // In order to receive the coordinates of all the points, we compute
    // their number, which is tdim times the number of points
    std::vector<int> nCoords(nPoints.size());
    std::transform(nPoints.cbegin(), nPoints.cend(), nCoords.begin(),
                   [tdim](const auto& i) { return tdim * i; });

    // The destination offsets of the upcoming allgatherv operation
    // are the partial sum of the number of coordinates
    std::vector<int> displacements(nProcs);
    displacements[0] = 0;
    std::partial_sum(nCoords.cbegin(), std::prev(nCoords.cend()),
                     std::next(displacements.begin()));

    // All the coordinates will be stored in this variable
    xt::xtensor<double, 2> globalX(
        xt::shape({std::accumulate(nPoints.cbegin(), nPoints.cend(), 0), 3}),
        0);

    MPI_Allgatherv(x_t.data(), x_t.size(), MPI_DOUBLE, globalX.data(),
                   nCoords.data(), displacements.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Each process will now check at which points it can evaluate
    // the interpolating function, and note that down in evaluationCells
    std::vector<std::int32_t> evaluationCells(globalX.shape(0), -1);

    auto connectivity = v.function_space()->mesh()->geometry().dofmap();

    // This BBT is useful for fast lookup of which cell contains a given point
    dolfinx::geometry::BoundingBoxTree bbt(*v.function_space()->mesh(), tdim);

    auto xv = v.function_space()->mesh()->geometry().x();
    // For each point at which the source function needs to be evaluated
    for (decltype(globalX.shape(0)) i = 0; i < globalX.shape(0); ++i)
    {
      // Get its coordinates
      double xp = globalX(i, 0);
      double yp = globalX(i, 1);
      double zp = globalX(i, 2);

      // For each cell that might contain that point
      for (const auto& j :
           dolfinx::geometry::compute_collisions(bbt, {xp, yp, zp}))
      {
        // Get the vertexes that belong to that cell
        auto vtx = connectivity.links(j);

        auto x1 = xv[3 * vtx[0]];
        auto y1 = xv[3 * vtx[0] + 1];
        auto z1 = xv[3 * vtx[0] + 2];
        auto x2 = xv[3 * vtx[1]];
        auto y2 = xv[3 * vtx[1] + 1];
        auto z2 = xv[3 * vtx[1] + 2];
        auto x3 = xv[3 * vtx[2]];
        auto y3 = xv[3 * vtx[2] + 1];
        auto z3 = xv[3 * vtx[2] + 2];
        auto x4 = xv[3 * vtx[3]];
        auto y4 = xv[3 * vtx[3] + 1];
        auto z4 = xv[3 * vtx[3] + 2];

        // We compute the barycentric coordinates of the given point
        // in the cell at hand
        xt::xarray<double> A = {{x1 - x4, x2 - x4, x3 - x4},
                                {y1 - y4, y2 - y4, y3 - y4},
                                {z1 - z4, z2 - z4, z3 - z4}};
        xt::xarray<double> b = {xp - x4, yp - y4, zp - z4};

        xt::xarray<double> l = xt::linalg::solve(A, b);

        // The point belongs to the cell only if all its barycentric
        // coordinates are positive. In this case
        if (l[0] >= 0 and l[1] >= 0 and l[2] >= 0
            and 1 - l[0] - l[1] - l[2] >= 0)
        {
          // Note that the cell can be used for interpolation
          evaluationCells[i] = j;
          // Do not look any further
          break;
        }
      }
    }

    // We now evaluate the interpolating function at all points
    xt::xtensor<T, 2> u_vals(
        xt::shape({globalX.shape(0),
                   static_cast<decltype(globalX.shape(0))>(
                       u.function_space()->element()->value_size())}),
        0);
    v.eval(globalX, evaluationCells, u_vals);

    // We need [do we?] a separate variable to store the reduced values
    auto finalU = xt::zeros_like(u_vals);

    // Since some points appear more than once in globalX, we cannot simply
    // reduce with a sum as the values at those points would be doubled.
    // Here we define a custom reduction operator that, given two values,
    // returns their sum if at least one of them is zero, and does nothing
    // otherwise
    MPI_Op MPI_SINGLESUM;
    MPI_Op_create(&SINGLESUM, true, &MPI_SINGLESUM);

    MPI_Allreduce(u_vals.data(), finalU.data(), u_vals.size(), MPI_TYPE<T>,
                  MPI_SINGLESUM, MPI_COMM_WORLD);

    // Now that each process has all the values, each process can extract
    // the portion that it needs
    std::vector<std::size_t> shape
        = {x.shape(1), static_cast<decltype(globalX.shape(0))>(
                           u.function_space()->element()->value_size())};
    xt::xarray<T> myU(shape);
    std::copy_n(finalU.cbegin()
                    + displacements[mpi_rank] / tdim
                          * u.function_space()->element()->value_size(),
                nPoints[mpi_rank] * u.function_space()->element()->value_size(),
                myU.begin());

    // This transposition is a quick and dirty solution and should be avoided
    xt::xarray<T> myU_t = xt::zeros_like(xt::transpose(myU));
    for (decltype(myU.shape(1)) i = 0; i < myU.shape(1); ++i)
    {
      for (decltype(myU.shape(0)) j = 0; j < myU.shape(0); ++j)
      {
        myU_t(i, j) = myU(j, i);
      }
    }

    // Finally, interpolate using the computed values
    fem::interpolate(u, myU_t, x, cells);

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
          + " of function space(" + std::to_string(element->value_dimension(i))
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

  mesh->topology_mutable().create_entity_permutations();
  const std::vector<std::uint32_t>& cell_info
      = mesh->topology().get_cell_permutation_info();

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
  const std::size_t value_size = std::accumulate(
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
