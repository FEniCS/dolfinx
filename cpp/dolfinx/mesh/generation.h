// Copyright (C) 2005-2023 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "cell_types.h"
#include "utils.h"
#include <array>
#include <cfloat>
#include <concepts>
#include <cstddef>
#include <limits>
#include <mpi.h>
#include <vector>

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
/// intervals will be `n` and the total number of vertices will be `n +
/// 1`.
///
/// @param[in] comm MPI communicator to build the mesh on
/// @param[in] nx Number of cells
/// @param[in] p End points of the interval
/// @param[in] partitioner Partitioning function for distributing cells
/// across MPI ranks.
/// @return A mesh
template <std::floating_point T = double>
Mesh<T> create_interval(MPI_Comm comm, std::int64_t nx, std::array<double, 2> p,
                        CellPartitionFunction partitioner = nullptr)
{
  if (!partitioner and dolfinx::MPI::size(comm) > 1)
    partitioner = create_cell_partitioner();

  fem::CoordinateElement<T> element(CellType::interval, 1);
  std::vector<T> x;
  std::vector<std::int64_t> cells;
  if (dolfinx::MPI::rank(comm) == 0)
  {
    const T a = p[0];
    const T b = p[1];
    const T ab = (b - a) / static_cast<T>(nx);

    if (std::abs(a - b) < std::numeric_limits<double>::epsilon())
    {
      throw std::runtime_error(
          "Length of interval is zero. Check your dimensions.");
    }

    if (b < a)
    {
      throw std::runtime_error(
          "Interval length is negative. Check order of arguments.");
    }

    if (nx < 1)
      throw std::runtime_error(
          "Number of points on interval must be at least 1");

    // Create vertices
    x.resize(nx + 1);
    for (std::int64_t ix = 0; ix <= nx; ix++)
      x[ix] = a + ab * static_cast<T>(ix);

    // Create intervals
    cells.resize(nx * 2);
    for (std::int64_t ix = 0; ix < nx; ++ix)
      for (std::int64_t j = 0; j < 2; ++j)
        cells[2 * ix + j] = ix + j;

    return create_mesh(comm, MPI_COMM_SELF, cells, element, MPI_COMM_SELF, x,
                       {x.size(), 1}, partitioner);
  }
  else
  {
    return create_mesh(comm, MPI_COMM_NULL, {}, element, MPI_COMM_NULL, x,
                       {x.size(), 1}, partitioner);
  }
}

namespace impl
{
template <std::floating_point T>
std::vector<T> create_geom(MPI_Comm comm,
                           std::array<std::array<double, 3>, 2> p,
                           std::array<std::int64_t, 3> n)
{
  // Extract data
  const std::array<double, 3> p0 = p[0];
  const std::array<double, 3> p1 = p[1];
  std::int64_t nx = n[0];
  std::int64_t ny = n[1];
  std::int64_t nz = n[2];

  const std::int64_t n_points = (nx + 1) * (ny + 1) * (nz + 1);
  std::array range_p = dolfinx::MPI::local_range(
      dolfinx::MPI::rank(comm), n_points, dolfinx::MPI::size(comm));

  // Extract minimum and maximum coordinates
  const double x0 = std::min(p0[0], p1[0]);
  const double x1 = std::max(p0[0], p1[0]);
  const double y0 = std::min(p0[1], p1[1]);
  const double y1 = std::max(p0[1], p1[1]);
  const double z0 = std::min(p0[2], p1[2]);
  const double z1 = std::max(p0[2], p1[2]);

  const T a = x0;
  const T b = x1;
  const T ab = (b - a) / static_cast<T>(nx);
  const T c = y0;
  const T d = y1;
  const T cd = (d - c) / static_cast<T>(ny);
  const T e = z0;
  const T f = z1;
  const T ef = (f - e) / static_cast<T>(nz);

  if (std::abs(x0 - x1) < 2.0 * std::numeric_limits<double>::epsilon()
      or std::abs(y0 - y1) < 2.0 * std::numeric_limits<double>::epsilon()
      or std::abs(z0 - z1) < 2.0 * std::numeric_limits<double>::epsilon())
  {
    throw std::runtime_error(
        "Box seems to have zero width, height or depth. Check dimensions");
  }

  if (nx < 1 or ny < 1 or nz < 1)
  {
    throw std::runtime_error(
        "BoxMesh has non-positive number of vertices in some dimension");
  }

  std::vector<T> geom;
  geom.reserve((range_p[1] - range_p[0]) * 3);
  const std::int64_t sqxy = (nx + 1) * (ny + 1);
  for (std::int64_t v = range_p[0]; v < range_p[1]; ++v)
  {
    const std::int64_t iz = v / sqxy;
    const std::int64_t p = v % sqxy;
    const std::int64_t iy = p / (nx + 1);
    const std::int64_t ix = p % (nx + 1);
    const T z = e + ef * static_cast<T>(iz);
    const T y = c + cd * static_cast<T>(iy);
    const T x = a + ab * static_cast<T>(ix);
    geom.insert(geom.end(), {x, y, z});
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
  if (subcomm != MPI_COMM_NULL)
  {
    x = create_geom<T>(subcomm, p, n);

    const std::int64_t nx = n[0];
    const std::int64_t ny = n[1];
    const std::int64_t nz = n[2];
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
      cells.insert(cells.end(),
                   {v0, v1, v3, v7, v0, v1, v7, v5, v0, v5, v7, v4,
                    v0, v3, v2, v7, v0, v6, v4, v7, v0, v2, v6, v7});
    }
  }

  fem::CoordinateElement<T> element(CellType::tetrahedron, 1);
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
  if (subcomm != MPI_COMM_NULL)
  {
    x = create_geom<T>(subcomm, p, n);

    // Create cuboids
    const std::int64_t nx = n[0];
    const std::int64_t ny = n[1];
    const std::int64_t nz = n[2];
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
  }

  fem::CoordinateElement<T> element(CellType::hexahedron, 1);
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
  if (subcomm != MPI_COMM_NULL)
  {
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
  }

  fem::CoordinateElement<T> element(CellType::prism, 1);
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
  if (dolfinx::MPI::rank(comm) == 0)
  {
    const std::array<double, 2> p0 = p[0];
    const std::array<double, 2> p1 = p[1];

    const std::int64_t nx = n[0];
    const std::int64_t ny = n[1];

    // Extract minimum and maximum coordinates
    const T x0 = std::min(p0[0], p1[0]);
    const T x1 = std::max(p0[0], p1[0]);
    const T y0 = std::min(p0[1], p1[1]);
    const T y1 = std::max(p0[1], p1[1]);

    const T a = x0;
    const T b = x1;
    const T ab = (b - a) / static_cast<T>(nx);
    const T c = y0;
    const T d = y1;
    const T cd = (d - c) / static_cast<T>(ny);

    if (std::abs(x0 - x1) < std::numeric_limits<double>::epsilon()
        or std::abs(y0 - y1) < std::numeric_limits<double>::epsilon())
    {
      throw std::runtime_error("Rectangle seems to have zero width, height or "
                               "depth. Check dimensions");
    }

    if (nx < 1 or ny < 1)
    {
      throw std::runtime_error(
          "Rectangle has non-positive number of vertices in some dimension: "
          "number of vertices must be at least 1 in each dimension");
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
          x.insert(x.end(), {a + ab * (static_cast<T>(ix) + 0.5), x1});
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
          cells.insert(cells.end(), {v0, v1, vmid, v0, v2, vmid, v1, v3, vmid,
                                     v2, v3, vmid});
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
  else
  {
    return create_mesh(comm, MPI_COMM_NULL, cells, element, MPI_COMM_NULL, x,
                       {x.size() / 2, 2}, partitioner);
  }
}

template <std::floating_point T>
Mesh<T> build_quad(MPI_Comm comm, const std::array<std::array<double, 2>, 2> p,
                   std::array<std::int64_t, 2> n,
                   const CellPartitionFunction& partitioner)
{
  fem::CoordinateElement<T> element(CellType::quadrilateral, 1);
  std::vector<std::int64_t> cells;
  std::vector<T> x;
  if (dolfinx::MPI::rank(comm) == 0)
  {
    const std::int64_t nx = n[0];
    const std::int64_t ny = n[1];
    const T a = p[0][0];
    const T b = p[1][0];
    const T ab = (b - a) / static_cast<T>(nx);
    const T c = p[0][1];
    const T d = p[1][1];
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
        cells.insert(cells.end(), {i0 + iy, i0 + iy + 1, i0 + iy + ny + 1,
                                   i0 + iy + ny + 2});
      }
    }

    return create_mesh(comm, MPI_COMM_SELF, cells, element, MPI_COMM_SELF, x,
                       {x.size() / 2, 2}, partitioner);
  }
  else
  {
    return create_mesh(comm, MPI_COMM_NULL, cells, element, MPI_COMM_NULL, x,
                       {x.size() / 2, 2}, partitioner);
  }
}
} // namespace impl
} // namespace dolfinx::mesh
