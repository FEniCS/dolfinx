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
    for (decltype(candidates.size()) i = 0; i < candidates.size(); ++i)
    {
      const auto xp = x_t(i, 0);
      const auto yp = x_t(i, 1);
      const auto zp = x_t(i, 2);

      for (const auto& p : candidates[i])
      {
        pointsToSend[p].push_back(xp);
        pointsToSend[p].push_back(yp);
        pointsToSend[p].push_back(zp);
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

    //    if (mpi_rank == 0)
    //    {
    //      for (size_t i = 0; i < 9692 * 3; ++i)
    //      {
    //        pointsToSend[0][i] = 1;
    //      }
    //      for (size_t i = 0; i < 9659 * 3; ++i)
    //      {
    //        pointsToSend[1][i] = 2;
    //      }
    //      for (size_t i = 0; i < 9692 * 3; ++i)
    //      {
    //        pointsToSend[2][i] = 3;
    //      }
    //    }
    //    if (mpi_rank == 1)
    //    {
    //      for (size_t i = 0; i < 9134 * 3; ++i)
    //      {
    //        pointsToSend[0][i] = 4;
    //      }
    //      for (size_t i = 0; i < 9414 * 3; ++i)
    //      {
    //        pointsToSend[1][i] = 5;
    //      }
    //      for (size_t i = 0; i < 10136 * 3; ++i)
    //      {
    //        pointsToSend[2][i] = 6;
    //      }
    //    }
    //    if (mpi_rank == 2)
    //    {
    //      for (size_t i = 0; i < 7984 * 3; ++i)
    //      {
    //        pointsToSend[0][i] = 7;
    //      }
    //      for (size_t i = 0; i < 8741 * 3; ++i)
    //      {
    //        pointsToSend[1][i] = 8;
    //      }
    //      for (size_t i = 0; i < 10068 * 3; ++i)
    //      {
    //        pointsToSend[2][i] = 9;
    //      }
    //    }

    xt::xtensor<double, 2> pointsToReceive(
        xt::shape(
            {nPointsToReceive / 3, static_cast<decltype(nPointsToReceive)>(3)}),
        0);
    MPI_Win window;
    MPI_Win_create(pointsToReceive.data(), sizeof(double) * nPointsToReceive,
                   sizeof(double), MPI_INFO_NULL, MPI_COMM_WORLD, &window);
    MPI_Win_fence(0, window);

    for (size_t i = 0; i < nProcs; ++i)
    {
      //      if (i == mpi_rank)
      //      {
      //        std::copy(pointsToSend[i].cbegin(), pointsToSend[i].cend(),
      //                  std::next(pointsToReceive.begin(),
      //                  sendingOffsets[i]));
      //      }
      //      else
      //      {
      MPI_Put(pointsToSend[i].data(), pointsToSend[i].size(), MPI_DOUBLE, i,
              sendingOffsets[i], pointsToSend[i].size(), MPI_DOUBLE, window);
      //      }
    }

    //    for (int i = 0; i < pointsToReceive.shape(0); ++i)
    //    {
    //      LOG(INFO) << i << "\t\t" << pointsToReceive(i, 0) << "\t"
    //                << pointsToReceive(i, 1) << "\t" << pointsToReceive(i, 2);
    //    }

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

    if (tdim == 3)
    {
      // For each point at which the source function needs to be evaluated
      for (decltype(pointsToReceive.shape(0)) i = 0;
           i < pointsToReceive.shape(0); ++i)
      {
        // Get its coordinates
        const double xp = pointsToReceive(i, 0);
        const double yp = pointsToReceive(i, 1);
        const double zp = pointsToReceive(i, 2);

        // For each cell that might contain that point
        for (const auto& j :
             dolfinx::geometry::compute_collisions(bbt, {xp, yp, zp}))
        {
          // Get the vertexes that belong to that cell
          const auto vtx = connectivity.links(j);

          const auto x1 = xv[3 * vtx[0]];
          const auto y1 = xv[3 * vtx[0] + 1];
          const auto z1 = xv[3 * vtx[0] + 2];
          const auto x2 = xv[3 * vtx[1]];
          const auto y2 = xv[3 * vtx[1] + 1];
          const auto z2 = xv[3 * vtx[1] + 2];
          const auto x3 = xv[3 * vtx[2]];
          const auto y3 = xv[3 * vtx[2] + 1];
          const auto z3 = xv[3 * vtx[2] + 2];
          const auto x4 = xv[3 * vtx[3]];
          const auto y4 = xv[3 * vtx[3] + 1];
          const auto z4 = xv[3 * vtx[3] + 2];

          // We compute the barycentric coordinates of the given point
          // in the cell at hand
          const xt::xarray<double> A = {{x1 - x4, x2 - x4, x3 - x4},
                                        {y1 - y4, y2 - y4, y3 - y4},
                                        {z1 - z4, z2 - z4, z3 - z4}};
          const xt::xarray<double> b = {xp - x4, yp - y4, zp - z4};

          const xt::xarray<double> l = xt::linalg::solve(A, b);

          auto tol = -0.0001;

          // The point belongs to the cell only if all its barycentric
          // coordinates are positive. In this case
          if (l[0] >= tol and l[1] >= tol and l[2] >= tol
              and 1 - l[0] - l[1] - l[2] >= tol)
          {
            // Note that the cell can be used for interpolation
            evaluationCells[i] = j;
            // Do not look any further
            break;
          }
        }
      }
    }
    else if (tdim == 2)
    {
      // For each point at which the source function needs to be evaluated
      for (decltype(pointsToReceive.shape(0)) i = 0;
           i < pointsToReceive.shape(0); ++i)
      {
        // Get its coordinates
        const double xp = pointsToReceive(i, 0);
        const double yp = pointsToReceive(i, 1);

        // For each cell that might contain that point
        for (const auto& j :
             dolfinx::geometry::compute_collisions(bbt, {xp, yp, 0}))
        {
          // Get the vertexes that belong to that cell
          const auto vtx = connectivity.links(j);

          const auto x1 = xv[3 * vtx[0]];
          const auto y1 = xv[3 * vtx[0] + 1];
          const auto x2 = xv[3 * vtx[1]];
          const auto y2 = xv[3 * vtx[1] + 1];
          const auto x3 = xv[3 * vtx[2]];
          const auto y3 = xv[3 * vtx[2] + 1];

          // We compute the barycentric coordinates of the given point
          // in the cell at hand
          const auto detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
          const auto lambda1
              = ((y2 - y3) * (xp - x3) + (x3 - x2) * (yp - y3)) / detT;
          const auto lambda2
              = ((y3 - y1) * (xp - x3) + (x1 - x3) * (yp - y3)) / detT;

          // The point belongs to the cell only if all its barycentric
          // coordinates are positive. In this case
          if (lambda1 >= 0 and lambda2 >= 0 and 1 - lambda1 - lambda2 >= 0)
          {
            // Note that the cell can be used for interpolation
            evaluationCells[i] = j;
            // Do not look any further
            break;
          }
        }
      }
    }
    else
    {
      throw std::runtime_error(
          "Interpolation not implemented for topological dimension "
          + std::to_string(tdim));
    }

    auto value_size = u.function_space()->element()->value_size();

    xt::xtensor<T, 2> values(
        xt::shape(
            {pointsToReceive.shape(0),
             static_cast<decltype(pointsToReceive.shape(0))>(value_size)}),
        0);
    v.eval(pointsToReceive, evaluationCells, values);

    //    for (decltype(pointsToReceive.shape(0)) i = 0; i <
    //    pointsToReceive.shape(0);
    //         ++i)
    //    {
    //      auto xp = pointsToReceive(i, 0);
    //      auto yp = pointsToReceive(i, 1);
    //      auto zp = pointsToReceive(i, 2);
    //      //          values(i, 0) = std::cos(10 * xp) * std::sin(10 * zp);
    //      values(i, 0) = mpi_rank + 1;
    //    }

    //    if (mpi_rank == 0)
    //    {
    //      for (size_t i = 0; i < 9692; ++i)
    //      {
    //        values(i, 0) = 1;
    //      }
    //      for (size_t i = 0; i < 9134; ++i)
    //      {
    //        values(9692 + i, 0) = 2;
    //      }
    //      for (size_t i = 0; i < 7984; ++i)
    //      {
    //        values(9692 + 9134 + i, 0) = 3;
    //      }
    //    }
    //    if (mpi_rank == 1)
    //    {
    //      for (size_t i = 0; i < 9659; ++i)
    //      {
    //        values(i, 0) = 4;
    //      }
    //      for (size_t i = 0; i < 9414; ++i)
    //      {
    //        values(9659 + i, 0) = 5;
    //      }
    //      for (size_t i = 0; i < 8741; ++i)
    //      {
    //        values(9659 + 9414 + i, 0) = 6;
    //      }
    //    }
    //    if (mpi_rank == 2)
    //    {
    //      for (size_t i = 0; i < 9692; ++i)
    //      {
    //        values(i, 0) = 7;
    //      }
    //      for (size_t i = 0; i < 10136; ++i)
    //      {
    //        values(9692 + i, 0) = 8;
    //      }
    //      for (size_t i = 0; i < 10068; ++i)
    //      {
    //        values(9692 + 10136 + i, 0) = 9;
    //      }
    //    }

    //    for (int i = 0; i < values.shape(0); ++i)
    //    {
    //      LOG(INFO) << values(i, 0);
    //    }

    const size_t sx
        = std::accumulate(nPointsToSend.cbegin(), nPointsToSend.cend(),
                          static_cast<size_t>(0))
          / 3 * value_size;
    xt::xtensor<T, 2> valuesToRetrieve(
        xt::shape({sx, static_cast<size_t>(value_size)}), 0);

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

    for (size_t i = 0; i < nProcs; ++i)
    {
      //      if (i == mpi_rank)
      //      {
      //        std::copy_n(values.cbegin()+ sendingOffsets[i] / 3,
      //                    pointsToSend[i].size() / 3 * value_size,
      //                    valuesToRetrieve.begin() + retrievingOffsets[i] *
      //                    value_size);
      //      }
      //      else
      //      {
      MPI_Get(valuesToRetrieve.data() + retrievingOffsets[i],
              pointsToSend[i].size() / 3 * value_size, MPI_TYPE<T>, i,
              sendingOffsets[i] / 3 * value_size,
              pointsToSend[i].size() / 3 * value_size, MPI_TYPE<T>, window);
      //      }
    }

    MPI_Win_fence(0, window);
    MPI_Win_free(&window);

    //    for (int i = 0; i < valuesToRetrieve.shape(0); ++i)
    //    {
    //      LOG(INFO) << i << "\t\t" << valuesToRetrieve(i, 0);
    //    }

    std::vector<std::size_t> s
        = {x_t.shape(0), static_cast<decltype(x_t.shape(0))>(value_size)};
    xt::xarray<T> myVals(s, static_cast<T>(0));

    std::vector<size_t> scanningProgress(nProcs, 0);

    for (decltype(candidates.size()) i = 0; i < candidates.size(); ++i)
    {
      for (const auto& p : candidates[i])
      {
        if (myVals(i, 0) == static_cast<T>(0))
        {
          myVals(i, 0)
              = valuesToRetrieve(retrievingOffsets[p] + scanningProgress[p], 0);
          //          LOG(INFO) << "myVals(" << i << ", 0) = " << myVals(i, 0);
        }
        ++scanningProgress[p];
      }
      //      LOG(INFO) << "";
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

    //
    //
    //
    //
    //
    //
    //
    //
    //
    // Gather a vector nPoints whose element i contains the number
    // of points needed by process i
    std::vector<int> nPoints(nProcs, -1);
    nPoints[mpi_rank] = x.shape(1);

    MPI_Allgather(nPoints.data() + mpi_rank, 1, MPI_INT, nPoints.data(), 1,
                  MPI_INT, MPI_COMM_WORLD);

    // In order to receive the coordinates of all the points, we compute
    // their number, which is tdim times the number of points
    std::vector<int> nCoords(nPoints.size());
    std::transform(nPoints.cbegin(), nPoints.cend(), nCoords.begin(),
                   [](const auto& i) { return 3 * i; });

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

    //    // Each process will now check at which points it can evaluate
    //    // the interpolating function, and note that down in evaluationCells
    //    std::vector<std::int32_t> evaluationCells(globalX.shape(0), -1);

    //    const auto connectivity =
    //    v.function_space()->mesh()->geometry().dofmap();

    //    // This BBT is useful for fast lookup of which cell contains a given
    //    point dolfinx::geometry::BoundingBoxTree
    //    bbt(*v.function_space()->mesh(), tdim,
    //                                           0.0001);

    //    const auto xv = v.function_space()->mesh()->geometry().x();

    //    if (tdim == 3)
    //    {
    //      // For each point at which the source function needs to be evaluated
    //      for (decltype(globalX.shape(0)) i = 0; i < globalX.shape(0); ++i)
    //      {
    //        // Get its coordinates
    //        const double xp = globalX(i, 0);
    //        const double yp = globalX(i, 1);
    //        const double zp = globalX(i, 2);

    //        // For each cell that might contain that point
    //        for (const auto& j :
    //             dolfinx::geometry::compute_collisions(bbt, {xp, yp, zp}))
    //        {
    //          // Get the vertexes that belong to that cell
    //          const auto vtx = connectivity.links(j);

    //          const auto x1 = xv[3 * vtx[0]];
    //          const auto y1 = xv[3 * vtx[0] + 1];
    //          const auto z1 = xv[3 * vtx[0] + 2];
    //          const auto x2 = xv[3 * vtx[1]];
    //          const auto y2 = xv[3 * vtx[1] + 1];
    //          const auto z2 = xv[3 * vtx[1] + 2];
    //          const auto x3 = xv[3 * vtx[2]];
    //          const auto y3 = xv[3 * vtx[2] + 1];
    //          const auto z3 = xv[3 * vtx[2] + 2];
    //          const auto x4 = xv[3 * vtx[3]];
    //          const auto y4 = xv[3 * vtx[3] + 1];
    //          const auto z4 = xv[3 * vtx[3] + 2];

    //          // We compute the barycentric coordinates of the given point
    //          // in the cell at hand
    //          const xt::xarray<double> A = {{x1 - x4, x2 - x4, x3 - x4},
    //                                        {y1 - y4, y2 - y4, y3 - y4},
    //                                        {z1 - z4, z2 - z4, z3 - z4}};
    //          const xt::xarray<double> b = {xp - x4, yp - y4, zp - z4};

    //          const xt::xarray<double> l = xt::linalg::solve(A, b);

    //          auto tol = -0.0001;

    //          // The point belongs to the cell only if all its barycentric
    //          // coordinates are positive. In this case
    //          if (l[0] >= tol and l[1] >= tol and l[2] >= tol
    //              and 1 - l[0] - l[1] - l[2] >= tol)
    //          {
    //            // Note that the cell can be used for interpolation
    //            evaluationCells[i] = j;
    //            // Do not look any further
    //            break;
    //          }
    //        }
    //      }
    //    }
    //    else if (tdim == 2)
    //    {
    //      // For each point at which the source function needs to be evaluated
    //      for (decltype(globalX.shape(0)) i = 0; i < globalX.shape(0); ++i)
    //      {
    //        // Get its coordinates
    //        const double xp = globalX(i, 0);
    //        const double yp = globalX(i, 1);

    //        // For each cell that might contain that point
    //        for (const auto& j :
    //             dolfinx::geometry::compute_collisions(bbt, {xp, yp, 0}))
    //        {
    //          // Get the vertexes that belong to that cell
    //          const auto vtx = connectivity.links(j);

    //          const auto x1 = xv[3 * vtx[0]];
    //          const auto y1 = xv[3 * vtx[0] + 1];
    //          const auto x2 = xv[3 * vtx[1]];
    //          const auto y2 = xv[3 * vtx[1] + 1];
    //          const auto x3 = xv[3 * vtx[2]];
    //          const auto y3 = xv[3 * vtx[2] + 1];

    //          // We compute the barycentric coordinates of the given point
    //          // in the cell at hand
    //          const auto detT = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3);
    //          const auto lambda1
    //              = ((y2 - y3) * (xp - x3) + (x3 - x2) * (yp - y3)) / detT;
    //          const auto lambda2
    //              = ((y3 - y1) * (xp - x3) + (x1 - x3) * (yp - y3)) / detT;

    //          // The point belongs to the cell only if all its barycentric
    //          // coordinates are positive. In this case
    //          if (lambda1 >= 0 and lambda2 >= 0 and 1 - lambda1 - lambda2 >=
    //          0)
    //          {
    //            // Note that the cell can be used for interpolation
    //            evaluationCells[i] = j;
    //            // Do not look any further
    //            break;
    //          }
    //        }
    //      }
    //    }
    //    else
    //    {
    //      throw std::runtime_error(
    //          "Interpolation not implemented for topological dimension "
    //          + std::to_string(tdim));
    //    }

    //    auto value_size = u.function_space()->element()->value_size();

    // We now evaluate the interpolating function at all points
    xt::xtensor<T, 2> u_vals(
        xt::shape({globalX.shape(0),
                   static_cast<decltype(globalX.shape(0))>(value_size)}),
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
        = {x.shape(1), static_cast<decltype(x.shape(0))>(value_size)};
    xt::xarray<T> myU(shape);
    std::copy_n(finalU.cbegin() + displacements[mpi_rank] / 3 * value_size,
                nPoints[mpi_rank] * value_size, myU.begin());

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
