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

namespace dolfinx::refinement
{

namespace impl
{

/// @brief Threshold marking helper
///
/// @param[in] marker Input marker
/// @param[in] threshold Lower bound for values to mark
///
/// @returns indices i which satisfy e_i > threshold.
template <std::floating_point T>
std::vector<std::int32_t> mark_threshold(std::span<const T> marker, T threshold)
{
  auto mark = [=](T e) { return e > threshold; };

  std::vector<std::int32_t> indices;
  indices.reserve(std::ranges::count_if(marker, mark));

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(marker.size()); ++i)
  {
    if (mark(marker[i]))
      indices.push_back(i);
  }

  return indices;
}

} // namespace impl

/// @brief Maximum marking of a marker.
///
/// @param[in] marker Input marker (local) - usually an error indicator per
/// entity
/// @param[in] theta Cut off parameter, 0 < θ < 1
/// @param[in] comm Communicator over which the maximum is computed.
/// @return Indices (local) of marker elements, which satisfy: marker_i ≥ θ
/// max(marker).
template <std::floating_point T>
std::vector<std::int32_t> mark_maximum(std::span<const T> marker, T theta,
                                       MPI_Comm comm)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("Theta needs to fullfill 0 < θ < 1.");

  T max = marker.empty() ? std::numeric_limits<T>::lowest()
                         : std::ranges::max(marker);
  MPI_Allreduce(MPI_IN_PLACE, &max, 1, dolfinx::MPI::mpi_t<T>, MPI_MAX, comm);

  auto indices = impl::mark_threshold<T>(marker, theta * max);

  spdlog::info("Marking (max) {} / {} (local) entities.", indices.size(),
               marker.size());

  return indices;
}

/// @brief Equidistribution marking of a marker.
///
/// @param[in] marker Input marker (local) - usually an error indicator per
/// entity
/// @param[in] theta Parameter, 0 < θ < 1
/// @param[in] comm Communicator over which the total marker is computed.
/// @return Indices (local) of marker elements, which satisfy: marker_i ≥ θ
/// (sum_i marker_i^2)^1/2 / N^1/2.
template <std::floating_point T>
std::vector<std::int32_t> mark_equidistribution(std::span<const T> marker,
                                                T theta, MPI_Comm comm)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("Theta needs to fullfill 0 < θ < 1.");

  T norm{0};
  for (T e : marker)
    norm += std::pow(e, 2);

  MPI_Allreduce(MPI_IN_PLACE, &norm, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, comm);

  norm = std::sqrt(norm);

  std::int32_t count = marker.size();
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, dolfinx::MPI::mpi_t<std::int32_t>,
                MPI_SUM, comm);

  auto indices = impl::mark_threshold<T>(marker, theta * norm / std::sqrt(count));

  spdlog::info("Marking (equi) {} / {} (local) entities.", indices.size(),
               marker.size());

  return indices;
}

} // namespace dolfinx::refinement
