// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/common/MPI.h>
#include <dolfinx/common/local_range.h>
#include <utility>
#include <vector>

namespace dolfinx::test
{
/// @brief Local slice of a structured cube of tetrahedra (six per cube)
/// and of the cube vertex coordinates.
///
/// @param[in] comm Communicator over which the cube is distributed.
/// @param[in] n Number of cubes along each axis.
/// @return Cell vertices, flattened, and vertex coordinates, flattened.
inline std::pair<std::vector<std::int64_t>, std::vector<double>>
cube(MPI_Comm comm, std::int64_t n)
{
  const int rank = dolfinx::MPI::rank(comm);
  const int size = dolfinx::MPI::size(comm);

  std::vector<std::int64_t> cells;
  std::array<std::int64_t, 2> rc = common::local_range(rank, n * n * n, size);
  for (std::int64_t i = rc[0]; i < rc[1]; ++i)
  {
    const std::int64_t iz = i / (n * n);
    const std::int64_t j = i % (n * n);
    const std::int64_t iy = j / n;
    const std::int64_t ix = j % n;
    const std::int64_t v0 = iz * (n + 1) * (n + 1) + iy * (n + 1) + ix;
    const std::int64_t v1 = v0 + 1;
    const std::int64_t v2 = v0 + (n + 1);
    const std::int64_t v3 = v1 + (n + 1);
    const std::int64_t v4 = v0 + (n + 1) * (n + 1);
    const std::int64_t v5 = v1 + (n + 1) * (n + 1);
    const std::int64_t v6 = v2 + (n + 1) * (n + 1);
    const std::int64_t v7 = v3 + (n + 1) * (n + 1);
    cells.insert(cells.end(), {v0, v1, v3, v7, v0, v1, v7, v5, v0, v5, v7, v4,
                               v0, v3, v2, v7, v0, v6, v4, v7, v0, v2, v6, v7});
  }

  std::vector<double> x;
  std::array<std::int64_t, 2> rp
      = common::local_range(rank, (n + 1) * (n + 1) * (n + 1), size);
  const std::int64_t sqxy = (n + 1) * (n + 1);
  for (std::int64_t v = rp[0]; v < rp[1]; ++v)
  {
    const std::int64_t p = v % sqxy;
    x.insert(x.end(), {static_cast<double>(p % (n + 1)) / n,
                       static_cast<double>(p / (n + 1)) / n,
                       static_cast<double>(v / sqxy) / n});
  }

  return {std::move(cells), std::move(x)};
}
} // namespace dolfinx::test
