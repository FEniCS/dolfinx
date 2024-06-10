// Copyright (C) 2005-2023 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <mpi.h>

#include "Mesh.h"
#include "cell_types.h"
#include "utils.h"

namespace dolfinx::mesh
{
/// Enum for different diagonal types
enum class DiagonalType
{
  left,
  right,
  crossed,
  shared_facet,
  left_right,
  right_left
};

namespace impl
{
template <std::floating_point T>
Mesh<T> build_tri(MPI_Comm comm, std::array<std::array<double, 2>, 2> p,
                  std::array<std::int64_t, 2> n,
                  const CellPartitionFunction& partitioner,
                  DiagonalType diagonal);

template <std::floating_point T>
Mesh<T> build_quad(MPI_Comm comm, const std::array<std::array<double, 2>, 2> p,
                   std::array<std::int64_t, 2> n,
                   const CellPartitionFunction& partitioner);

template <std::floating_point T>
std::vector<T> create_geom(MPI_Comm comm,
                           std::array<std::array<double, 3>, 2> p,
                           std::array<std::int64_t, 3> n);

template <std::floating_point T>
Mesh<T> build_tet(MPI_Comm comm, MPI_Comm subcomm,
                  std::array<std::array<double, 3>, 2> p,
                  std::array<std::int64_t, 3> n,
                  const CellPartitionFunction& partitioner);

template <std::floating_point T>
Mesh<T> build_hex(MPI_Comm comm, MPI_Comm subcomm,
                  std::array<std::array<double, 3>, 2> p,
                  std::array<std::int64_t, 3> n,
                  const CellPartitionFunction& partitioner);

template <std::floating_point T>
Mesh<T> build_prism(MPI_Comm comm, MPI_Comm subcomm,
                    std::array<std::array<double, 3>, 2> p,
                    std::array<std::int64_t, 3> n,
                    const CellPartitionFunction& partitioner);
} // namespace impl

/// @brief Create a uniform mesh::Mesh over rectangular prism spanned by
/// the two points `p`.
///
/// The order of the two points is not important in terms of minimum and
/// maximum coordinates. The total number of vertices will be `(n[0] +
/// 1)*(n[1] + 1)*(n[2] + 1)`. For tetrahedra there will be  will be
/// `6*n[0]*n[1]*n[2]` cells. For hexahedra the number of cells will be
/// `n[0]*n[1]*n[2]`.
///
/// @param[in] comm MPI communicator to distribute the mesh on.
/// @param[in] subcomm MPI communicator to construct and partition the
/// mesh topology on. If the process should not be involved in the
/// topology creation and partitioning then this communicator should be
/// `MPI_COMM_NULL`.
/// @param[in] p Corner of the box.
/// @param[in] n Number of cells in each direction.
/// @param[in] celltype Cell shape.
/// @param[in] partitioner Partitioning function for distributing cells
/// across MPI ranks.
/// @return Mesh
template <std::floating_point T = double>
Mesh<T> create_box(MPI_Comm comm, MPI_Comm subcomm,
                   std::array<std::array<double, 3>, 2> p,
                   std::array<std::int64_t, 3> n, CellType celltype,
                   CellPartitionFunction partitioner = nullptr)
{
  if (std::ranges::any_of(n, [](auto e) { return e < 1; }))
    throw std::runtime_error("At least one cell per dimension is required");

  for (int32_t i = 0; i < 3; i++)
  {
    if (p[0][i] >= p[1][i])
      throw std::runtime_error("It must hold p[0] < p[1].");
  }

  if (!partitioner and dolfinx::MPI::size(comm) > 1)
    partitioner = create_cell_partitioner();

  switch (celltype)
  {
  case CellType::tetrahedron:
    return impl::build_tet<T>(comm, subcomm, p, n, partitioner);
  case CellType::hexahedron:
    return impl::build_hex<T>(comm, subcomm, p, n, partitioner);
  case CellType::prism:
    return impl::build_prism<T>(comm, subcomm, p, n, partitioner);
  default:
    throw std::runtime_error("Generate box mesh. Wrong cell type");
  }
}

/// @brief Create a uniform mesh::Mesh over rectangular prism spanned by
/// the two points `p`.
///
/// The order of the two points is not important in terms of minimum and
/// maximum coordinates. The total number of vertices will be `(n[0] +
/// 1)*(n[1] + 1)*(n[2] + 1)`. For tetrahedra there will be  will be
/// `6*n[0]*n[1]*n[2]` cells. For hexahedra the number of cells will be
/// `n[0]*n[1]*n[2]`.
///
/// @param[in] comm MPI communicator to distribute the mesh on.
/// @param[in] p Corner of the box.
/// @param[in] n Number of cells in each direction.
/// @param[in] celltype Cell shape.
/// @param[in] partitioner Partitioning function for distributing cells
/// across MPI ranks.
/// @return Mesh
template <std::floating_point T = double>
Mesh<T> create_box(MPI_Comm comm, std::array<std::array<double, 3>, 2> p,
                   std::array<std::int64_t, 3> n, CellType celltype,
                   const CellPartitionFunction& partitioner = nullptr)
{
  return create_box<T>(comm, comm, p, n, celltype, partitioner);
}

/// @brief Create a uniform mesh::Mesh over the rectangle spanned by the
/// two points `p`.
///
/// The order of the two points is not important in terms of minimum and
/// maximum coordinates. The total number of vertices will be `(n[0] +
/// 1)*(n[1] + 1)`. For triangles there will be  will be `2*n[0]*n[1]`
/// cells. For quadrilaterals the number of cells will be `n[0]*n[1]`.
///
/// @param[in] comm MPI communicator to build the mesh on.
/// @param[in] p Bottom-left and top-right corners of the rectangle.
/// @param[in] n Number of cells in each direction.
/// @param[in] celltype Cell shape.
/// @param[in] partitioner Partitioning function for distributing cells
/// across MPI ranks.
/// @param[in] diagonal Direction of diagonals
/// @return Mesh
template <std::floating_point T = double>
Mesh<T> create_rectangle(MPI_Comm comm, std::array<std::array<double, 2>, 2> p,
                         std::array<std::int64_t, 2> n, CellType celltype,
                         CellPartitionFunction partitioner,
                         DiagonalType diagonal = DiagonalType::right)
{
  if (std::ranges::any_of(n, [](auto e) { return e < 1; }))
    throw std::runtime_error("At least one cell per dimension is required");

  for (int32_t i = 0; i < 2; i++)
  {
    if (p[0][i] >= p[1][i])
      throw std::runtime_error("It must hold p[0] < p[1].");
  }

  if (!partitioner and dolfinx::MPI::size(comm) > 1)
    partitioner = create_cell_partitioner();

  switch (celltype)
  {
  case CellType::triangle:
    return impl::build_tri<T>(comm, p, n, partitioner, diagonal);
  case CellType::quadrilateral:
    return impl::build_quad<T>(comm, p, n, partitioner);
  default:
    throw std::runtime_error("Generate rectangle mesh. Wrong cell type");
  }
}

/// @brief Create a uniform mesh::Mesh over the rectangle spanned by the
/// two points `p`.
///
/// The order of the two points is not important in terms of minimum and
/// maximum coordinates. The total number of vertices will be `(n[0] +
/// 1)*(n[1] + 1)`. For triangles there will be  will be `2*n[0]*n[1]`
/// cells. For quadrilaterals the number of cells will be `n[0]*n[1]`.
///
/// @param[in] comm MPI communicator to build the mesh on
/// @param[in] p Two corner points
/// @param[in] n Number of cells in each direction
/// @param[in] celltype Cell shape
/// @param[in] diagonal Direction of diagonals
/// @return Mesh
template <std::floating_point T = double>
Mesh<T> create_rectangle(MPI_Comm comm, std::array<std::array<double, 2>, 2> p,
                         std::array<std::int64_t, 2> n, CellType celltype,
                         DiagonalType diagonal = DiagonalType::right)
{
  return create_rectangle<T>(comm, p, n, celltype, nullptr, diagonal);
}

/// @brief Interval mesh of the 1D line `[a, b]`.
///
/// Given `n` cells in the axial direction, the total number of
/// intervals will be `n` and the total number of vertices will be
/// `n + 1`.
///
/// @param[in] comm MPI communicator to build the mesh on.
/// @param[in] n Number of cells.
/// @param[in] p End points of the interval.
/// @param[in] ghost_mode ghost mode of the created mesh, defaults to none
/// @param[in] partitioner Partitioning function for distributing cells
/// across MPI ranks.
/// @return A mesh.
template <std::floating_point T = double>
Mesh<T> create_interval(MPI_Comm comm, std::int64_t n, std::array<double, 2> p, mesh::GhostMode ghost_mode = mesh::GhostMode::none,
                        CellPartitionFunction partitioner = nullptr)
{
  if (n < 1)
    throw std::runtime_error("At least one cell is required.");

  const auto [a, b] = p;
  if (a >= b)
    throw std::runtime_error("It must hold p[0] < p[1].");

  if (!partitioner and dolfinx::MPI::size(comm) > 1)
    partitioner = create_cell_partitioner(ghost_mode);

  fem::CoordinateElement<T> element(CellType::interval, 1);
  std::vector<T> x;
  std::vector<std::int64_t> cells;

  if (dolfinx::MPI::rank(comm) != 0)
  {
    return create_mesh(comm, MPI_COMM_NULL, {}, element, MPI_COMM_NULL, x,
                       {x.size(), 1}, partitioner);
  }

  const T h = (b - a) / static_cast<T>(n);

  if (std::abs(a - b) < std::numeric_limits<T>::epsilon())
  {
    throw std::runtime_error(
        "Length of interval is zero. Check your dimensions.");
  }

  // Create vertices
  x.reserve(n + 1);
  for (std::int64_t idx = 0; idx < n + 1; idx++)
    x.emplace_back(a + h * static_cast<T>(idx));

  // Create intervals -> cells=[0,1,1,...,n-1,n-1,n]
  cells.reserve(n * 2);
  for (std::int64_t idx = 0; idx < n * 2; idx++)
    cells.emplace_back(std::floor(idx / 2) + idx % 2);

  return create_mesh(comm, MPI_COMM_SELF, cells, element, MPI_COMM_SELF, x,
                     {x.size(), 1}, partitioner);
}

namespace impl
{
template <std::floating_point T>
std::vector<T> create_geom(MPI_Comm comm,
                           std::array<std::array<double, 3>, 2> p,
                           std::array<std::int64_t, 3> n)
{
  // Extract data
  auto [p0, p1] = p;
  const auto [nx, ny, nz] = n;

  assert(std::ranges::all_of(n, [](auto e) { return e >= 1; }));
  for (std::int64_t i = 0; i < 3; i++)
    assert(p0[i] < p1[i]);

  // structured grid cuboid extents
  const std::array<T, 3> extents = {
      (p1[0] - p0[0]) / static_cast<T>(nx),
      (p1[1] - p0[1]) / static_cast<T>(ny),
      (p1[2] - p0[2]) / static_cast<T>(nz),
  };

  if (std::ranges::any_of(
          extents, [](auto e)
          { return std::abs(e) < 2.0 * std::numeric_limits<T>::epsilon(); }))
  {
    throw std::runtime_error(
        "Box seems to have zero width, height or depth. Check dimensions");
  }

  const std::int64_t n_points = (nx + 1) * (ny + 1) * (nz + 1);
  const auto [range_begin, range_end] = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_points, dolfinx::MPI::size(comm));

  std::vector<T> geom;
  geom.reserve((range_end - range_begin) * 3);
  const std::int64_t sqxy = (nx + 1) * (ny + 1);
  for (std::int64_t v = range_begin; v < range_end; ++v)
  {
    // lexiographic index to spacial index
    const std::int64_t p = v % sqxy;
    std::array<std::int64_t, 3> idx{p % (nx + 1), p / (nx + 1), v / sqxy};

    // vertex = p0 + idx * extents (elementwise)
    for (std::size_t i = 0; i < idx.size(); i++)
      geom.emplace_back(p0[i] + static_cast<T>(idx[i]) * extents[i]);
  }

  return geom;
}

template <std::floating_point T>
Mesh<T> build_tet(MPI_Comm comm, MPI_Comm subcomm,
                  std::array<std::array<double, 3>, 2> p,
                  std::array<std::int64_t, 3> n,
                  const CellPartitionFunction& partitioner)
{
  common::Timer timer("Build BoxMesh (tetrahedra)");
  std::vector<T> x;
  std::vector<std::int64_t> cells;
  fem::CoordinateElement<T> element(CellType::tetrahedron, 1);

  if (subcomm == MPI_COMM_NULL)
    return create_mesh(comm, subcomm, cells, element, subcomm, x,
                       {x.size() / 3, 3}, partitioner);

  x = create_geom<T>(subcomm, p, n);

  const auto [nx, ny, nz] = n;
  const std::int64_t n_cells = nx * ny * nz;

  std::array range_c = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(subcomm), n_cells, dolfinx::MPI::size(subcomm));
  cells.reserve(6 * (range_c[1] - range_c[0]) * 4);

  // Create tetrahedra
  for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
  {
    const std::int64_t iz = i / (nx * ny);
    const std::int64_t j = i % (nx * ny);
    const std::int64_t iy = j / nx;
    const std::int64_t ix = j % nx;
    const std::int64_t v0 = iz * (nx + 1) * (ny + 1) + iy * (nx + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (nx + 1);
    const std::int64_t v3 = v1 + (nx + 1);
    const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
    const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
    const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
    const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);

    // Note that v0 < v1 < v2 < v3 < vmid
    cells.insert(cells.end(), {v0, v1, v3, v7, v0, v1, v7, v5, v0, v5, v7, v4,
                               v0, v3, v2, v7, v0, v6, v4, v7, v0, v2, v6, v7});
  }

  return create_mesh(comm, subcomm, cells, element, subcomm, x,
                     {x.size() / 3, 3}, partitioner);
}

template <std::floating_point T>
mesh::Mesh<T> build_hex(MPI_Comm comm, MPI_Comm subcomm,
                        std::array<std::array<double, 3>, 2> p,
                        std::array<std::int64_t, 3> n,
                        const CellPartitionFunction& partitioner)
{
  common::Timer timer("Build BoxMesh (hexahedra)");
  std::vector<T> x;
  std::vector<std::int64_t> cells;
  fem::CoordinateElement<T> element(CellType::hexahedron, 1);

  if (subcomm == MPI_COMM_NULL)
    return create_mesh(comm, subcomm, cells, element, subcomm, x,
                       {x.size() / 3, 3}, partitioner);

  x = create_geom<T>(subcomm, p, n);

  // Create cuboids
  const auto [nx, ny, nz] = n;
  const std::int64_t n_cells = nx * ny * nz;
  std::array range_c = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(subcomm), n_cells, dolfinx::MPI::size(subcomm));
  cells.reserve((range_c[1] - range_c[0]) * 8);
  for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
  {
    const std::int64_t iz = i / (nx * ny);
    const std::int64_t j = i % (nx * ny);
    const std::int64_t iy = j / nx;
    const std::int64_t ix = j % nx;

    const std::int64_t v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (nx + 1);
    const std::int64_t v3 = v1 + (nx + 1);
    const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
    const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
    const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
    const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);
    cells.insert(cells.end(), {v0, v1, v2, v3, v4, v5, v6, v7});
  }

  return create_mesh(comm, subcomm, cells, element, subcomm, x,
                     {x.size() / 3, 3}, partitioner);
}

template <std::floating_point T>
Mesh<T> build_prism(MPI_Comm comm, MPI_Comm subcomm,
                    std::array<std::array<double, 3>, 2> p,
                    std::array<std::int64_t, 3> n,
                    const CellPartitionFunction& partitioner)
{
  std::vector<T> x;
  std::vector<std::int64_t> cells;
  fem::CoordinateElement<T> element(CellType::prism, 1);

  if (subcomm == MPI_COMM_NULL)
    return create_mesh(comm, subcomm, cells, element, subcomm, x,
                       {x.size() / 3, 3}, partitioner);

  x = create_geom<T>(subcomm, p, n);

  const std::int64_t nx = n[0];
  const std::int64_t ny = n[1];
  const std::int64_t nz = n[2];
  const std::int64_t n_cells = nx * ny * nz;
  std::array range_c = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_cells, dolfinx::MPI::size(comm));
  const std::int64_t cell_range = range_c[1] - range_c[0];

  // Create cuboids

  cells.reserve(2 * cell_range * 6);
  for (std::int64_t i = range_c[0]; i < range_c[1]; ++i)
  {
    const std::int64_t iz = i / (nx * ny);
    const std::int64_t j = i % (nx * ny);
    const std::int64_t iy = j / nx;
    const std::int64_t ix = j % nx;

    const std::int64_t v0 = (iz * (ny + 1) + iy) * (nx + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (nx + 1);
    const std::int64_t v3 = v1 + (nx + 1);
    const std::int64_t v4 = v0 + (nx + 1) * (ny + 1);
    const std::int64_t v5 = v1 + (nx + 1) * (ny + 1);
    const std::int64_t v6 = v2 + (nx + 1) * (ny + 1);
    const std::int64_t v7 = v3 + (nx + 1) * (ny + 1);
    cells.insert(cells.end(), {v0, v1, v2, v4, v5, v6});
    cells.insert(cells.end(), {v1, v2, v3, v5, v6, v7});
  }

  return create_mesh(comm, subcomm, cells, element, subcomm, x,
                     {x.size() / 3, 3}, partitioner);
}

template <std::floating_point T>
Mesh<T> build_tri(MPI_Comm comm, std::array<std::array<double, 2>, 2> p,
                  std::array<std::int64_t, 2> n,
                  const CellPartitionFunction& partitioner,
                  DiagonalType diagonal)
{
  fem::CoordinateElement<T> element(CellType::triangle, 1);
  std::vector<T> x;
  std::vector<std::int64_t> cells;

  if (dolfinx::MPI::rank(comm) != 0)
    return create_mesh(comm, MPI_COMM_NULL, cells, element, MPI_COMM_NULL, x,
                       {x.size() / 2, 2}, partitioner);

  const auto [p0, p1] = p;
  const auto [nx, ny] = n;

  const auto [a, c] = p0;
  const auto [b, d] = p1;

  const T ab = (b - a) / static_cast<T>(nx);
  const T cd = (d - c) / static_cast<T>(ny);

  if (std::abs(b - a) < std::numeric_limits<T>::epsilon()
      or std::abs(d - c) < std::numeric_limits<T>::epsilon())
  {
    throw std::runtime_error("Rectangle seems to have zero width, height or "
                             "depth. Check dimensions");
  }

  // Create vertices and cells
  std::int64_t nv, nc;
  switch (diagonal)
  {
  case DiagonalType::crossed:
    nv = (nx + 1) * (ny + 1) + nx * ny;
    nc = 4 * nx * ny;
    break;
  default:
    nv = (nx + 1) * (ny + 1);
    nc = 2 * nx * ny;
  }

  x.reserve(nv * 2);
  cells.reserve(nc * 3);

  // Create main vertices
  std::int64_t vertex = 0;
  for (std::int64_t iy = 0; iy <= ny; iy++)
  {
    const T x1 = c + cd * static_cast<T>(iy);
    for (std::int64_t ix = 0; ix <= nx; ix++)
      x.insert(x.end(), {a + ab * static_cast<T>(ix), x1});
  }

  // Create midpoint vertices if the mesh type is crossed
  switch (diagonal)
  {
  case DiagonalType::crossed:
    for (std::int64_t iy = 0; iy < ny; iy++)
    {
      const T x1 = c + cd * (static_cast<T>(iy) + 0.5);
      for (std::int64_t ix = 0; ix < nx; ix++)
      {
        const T x0 = a + ab * (static_cast<T>(ix) + 0.5);
        x.insert(x.end(), {x0, x1});
      }
    }
    break;
  default:
    break;
  }

  // Create triangles
  switch (diagonal)
  {
  case DiagonalType::crossed:
  {
    for (std::int64_t iy = 0; iy < ny; iy++)
    {
      for (std::int64_t ix = 0; ix < nx; ix++)
      {
        const std::int64_t v0 = iy * (nx + 1) + ix;
        const std::int64_t v1 = v0 + 1;
        const std::int64_t v2 = v0 + (nx + 1);
        const std::int64_t v3 = v1 + (nx + 1);
        const std::int64_t vmid = (nx + 1) * (ny + 1) + iy * nx + ix;

        // Note that v0 < v1 < v2 < v3 < vmid
        cells.insert(cells.end(),
                     {v0, v1, vmid, v0, v2, vmid, v1, v3, vmid, v2, v3, vmid});
      }
    }
    break;
  }
  default:
  {
    DiagonalType local_diagonal = diagonal;
    for (std::int64_t iy = 0; iy < ny; iy++)
    {
      // Set up alternating diagonal
      switch (diagonal)
      {
      case DiagonalType::right_left:
        if (iy % 2)
          local_diagonal = DiagonalType::right;
        else
          local_diagonal = DiagonalType::left;
        break;
      case DiagonalType::left_right:
        if (iy % 2)
          local_diagonal = DiagonalType::left;
        else
          local_diagonal = DiagonalType::right;
        break;
      default:
        break;
      }
      for (std::int64_t ix = 0; ix < nx; ix++)
      {
        const std::int64_t v0 = iy * (nx + 1) + ix;
        const std::int64_t v1 = v0 + 1;
        const std::int64_t v2 = v0 + (nx + 1);
        const std::int64_t v3 = v1 + (nx + 1);

        switch (local_diagonal)
        {
        case DiagonalType::left:
        {
          cells.insert(cells.end(), {v0, v1, v2, v1, v2, v3});
          if (diagonal == DiagonalType::right_left
              or diagonal == DiagonalType::left_right)
          {
            local_diagonal = DiagonalType::right;
          }
          break;
        }
        default:
        {
          cells.insert(cells.end(), {v0, v1, v3, v0, v2, v3});
          if (diagonal == DiagonalType::right_left
              or diagonal == DiagonalType::left_right)
          {
            local_diagonal = DiagonalType::left;
          }
        }
        }
      }
    }
  }
  }

  return create_mesh(comm, MPI_COMM_SELF, cells, element, MPI_COMM_SELF, x,
                     {x.size() / 2, 2}, partitioner);
}

template <std::floating_point T>
Mesh<T> build_quad(MPI_Comm comm, const std::array<std::array<double, 2>, 2> p,
                   std::array<std::int64_t, 2> n,
                   const CellPartitionFunction& partitioner)
{
  fem::CoordinateElement<T> element(CellType::quadrilateral, 1);
  std::vector<std::int64_t> cells;
  std::vector<T> x;

  if (dolfinx::MPI::rank(comm) != 0)
    return create_mesh(comm, MPI_COMM_NULL, cells, element, MPI_COMM_NULL, x,
                       {x.size() / 2, 2}, partitioner);

  const auto [nx, ny] = n;
  const auto [a, c] = p[0];
  const auto [b, d] = p[1];

  const T ab = (b - a) / static_cast<T>(nx);
  const T cd = (d - c) / static_cast<T>(ny);

  // Create vertices
  x.reserve((nx + 1) * (ny + 1) * 2);
  std::int64_t vertex = 0;
  for (std::int64_t ix = 0; ix <= nx; ix++)
  {
    T x0 = a + ab * static_cast<T>(ix);
    for (std::int64_t iy = 0; iy <= ny; iy++)
      x.insert(x.end(), {x0, c + cd * static_cast<T>(iy)});
  }

  // Create rectangles
  cells.reserve(nx * ny * 4);
  for (std::int64_t ix = 0; ix < nx; ix++)
  {
    for (std::int64_t iy = 0; iy < ny; iy++)
    {
      std::int64_t i0 = ix * (ny + 1);
      cells.insert(cells.end(),
                   {i0 + iy, i0 + iy + 1, i0 + iy + ny + 1, i0 + iy + ny + 2});
    }
  }

  return create_mesh(comm, MPI_COMM_SELF, cells, element, MPI_COMM_SELF, x,
                     {x.size() / 2, 2}, partitioner);
}
} // namespace impl
} // namespace dolfinx::mesh
