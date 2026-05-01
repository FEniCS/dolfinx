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
/// @param[in] marker Input marker \f$ \eta \f$
/// @param[in] threshold Lower bound for values to mark
///
/// @returns indices \f$ i \f$ which satisfy \f$ \eta_i > \text{threshold} \f$.
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
/// @param[in] marker Input marker (local) \f$ \eta \f$ - usually an error
/// indicator per entity
/// @param[in] theta Cut off parameter, \f$ 0 < \theta < 1 \f$
/// @param[in] comm Communicator over which the maximum is computed.
/// @return Indices (local) of marker elements, which satisfy: \f$
/// \eta_i \geq \theta \max_j \eta_j \f$.
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
/// @param[in] marker Input marker (local) \f$ \eta \f$ - usually an error
/// indicator per entity
/// @param[in] theta Parameter, \f$ 0 < \theta < 1 \f$
/// @param[in] comm Communicator over which the total marker is computed.
/// @return Local indices of marked entities, which satisfy: \f$
/// \eta_i \geq \theta \frac{|\eta|_2}{\sqrt{N}} \f$.
template <std::floating_point T>
std::vector<std::int32_t> mark_equidistribution(std::span<const T> marker,
                                                T theta, MPI_Comm comm)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("Theta needs to fullfill 0 < θ < 1.");

  auto norm
      = std::inner_product(marker.begin(), marker.end(), marker.begin(), T{0});

  MPI_Allreduce(MPI_IN_PLACE, &norm, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, comm);

  norm = std::sqrt(norm);

  std::int32_t count = marker.size();
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, dolfinx::MPI::mpi_t<std::int32_t>,
                MPI_SUM, comm);

  auto indices
      = impl::mark_threshold<T>(marker, theta * norm / std::sqrt(count));

  spdlog::info("Marking (equi) {} / {} (local) entities.", indices.size(),
               marker.size());

  return indices;
}

} // namespace dolfinx::refinement
