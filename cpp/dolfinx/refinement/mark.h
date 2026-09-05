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
#include <limits>
#include <mpi.h>
#include <spdlog/spdlog.h>
#include <vector>

#include "dolfinx/common/MPI.h"
#include "dolfinx/la/Vector.h"

namespace dolfinx::refinement
{

/// @brief Maximum marking of a marker.
///
/// @param[in] marker Input marker - usually an error indicator per
/// entity
/// @param[in] theta Cut off parameter, 0 < θ < 1
/// @return Indices (local, including ghosts) of marker elements, which satisfy:
/// marker_i ≥ θ max(marker).
template <std::floating_point T>
std::vector<std::int32_t> mark_maximum(const dolfinx::la::Vector<T>& marker,
                                       T theta)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("Theta needs to fulfill 0 < θ < 1.");

  auto im = marker.index_map();
  MPI_Comm comm = im->comm();

  std::span local_part = std::span{marker.array()}.subspan(0, im->size_local());

  T max = local_part.empty() ? std::numeric_limits<T>::lowest()
                             : std::ranges::max(local_part);
  MPI_Allreduce(MPI_IN_PLACE, &max, 1, dolfinx::MPI::mpi_t<T>, MPI_MAX, comm);

  auto mark = [&theta, &max](T e) { return e > theta * max; };

  std::vector<std::int32_t> indices;
  indices.reserve(std::ranges::count_if(marker.array(), mark));

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(marker.array().size());
       ++i)
  {
    if (mark(marker.array()[i]))
      indices.push_back(i);
  }

  spdlog::info("Marking (max) {} / {} (local) entities.", indices.size(),
               im->size_local() + im->num_ghosts());

  return indices;
}

} // namespace dolfinx::refinement
