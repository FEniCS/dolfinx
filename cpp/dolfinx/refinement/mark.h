// Copyright (C) 2026 Paul T. Kühner
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mpi.h>
#include <spdlog/spdlog.h>
#include <vector>

#include "dolfinx/common/MPI.h"

namespace dolfinx::refinement
{

/// @brief Maximum marking of a marker.
///
/// @param[in] marker Input marker (local) - usually an error indicator per
/// entity
/// @param[in] theta Cut off parameter, 0 ≤ θ ≤ 1
/// @param[in] comm Communicator over which the maximum is computed.
/// @return Indices (local) of marker elements, which satisfy: marker_i ≥ θ
/// max(marker).
template <std::floating_point T>
std::vector<std::int32_t> mark_maximum(std::span<const T> marker, T theta,
                                       MPI_Comm comm)
{
  assert((0 <= theta) && (theta <= 1));

  T max = marker.empty() ? std::numeric_limits<T>::min()
                         : std::ranges::max(marker);
  MPI_Allreduce(MPI_IN_PLACE, &max, 1, dolfinx::MPI::mpi_t<T>, MPI_MAX, comm);

  auto mark = [=](auto e) { return e >= theta * max; };

  std::vector<std::int32_t> indices;
  indices.reserve(std::ranges::count_if(marker, mark));

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(marker.size()); ++i)
  {
    if (mark(marker[i]))
      indices.push_back(i);
  }

  spdlog::info("Marking (max) {} / {} (local) entities.", indices.size(),
               marker.size());

  return indices;
}

} // namespace dolfinx::refinement
