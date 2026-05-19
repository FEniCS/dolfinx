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

/// @brief Computes local threshold-based marking.
///
/// Helper for other marking routines.
///
/// @param[in] indicators Entity-wise indicator \f$ \eta \f$ used for marking.
/// @param[in] threshold Threshold value; indicators greater than this are
/// marked.
///
/// @returns list of indices \f$ i \f$ of indicator entries that satisfy
/// \f$ \eta_i > \text{threshold} \f$.
template <std::floating_point T>
std::vector<std::int32_t> mark_threshold(std::span<const T> indicators,
                                         T threshold)
{
  auto mark = [threshold](T e) { return e > threshold; };

  std::vector<std::int32_t> indices;
  indices.reserve(std::ranges::count_if(indicators, mark));

  for (std::int32_t i = 0; i < static_cast<std::int32_t>(indicators.size());
       ++i)
  {
    if (mark(indicators[i]))
      indices.push_back(i);
  }

  return indices;
}

} // namespace impl

/// @brief Computes maximum-based marking of an indicator.
///
/// Returns the indices \f$ i \f$ of the indicators $eta_i$ that satisfy the
/// maximum threshold: \f$ \eta_i \geq \theta \max_j \eta_j \f$.
///
/// @param[in] comm Communicator to compute the maximum over.
/// @param[in] indicators Indicators (local) \f$ \eta_i \f$ -
/// usually an error indicator associated with mesh entity \f$ i \f$.
/// @param[in] theta Parameter, \f$ 0 < \theta < 1 \f$.
/// @return Indices (local) of marker elements, that satisfy the maximum
/// threshold:
template <std::floating_point T>
std::vector<std::int32_t> mark_maximum(MPI_Comm comm,
                                       std::span<const T> indicators, T theta)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("Theta needs to fullfill 0 < θ < 1.");

  T max = indicators.empty() ? std::numeric_limits<T>::lowest()
                             : std::ranges::max(indicators);
  MPI_Allreduce(MPI_IN_PLACE, &max, 1, dolfinx::MPI::mpi_t<T>, MPI_MAX, comm);

  auto indices = impl::mark_threshold<T>(indicators, theta * max);

  spdlog::info("Marking (maximum) {} / {} (local) entities.", indices.size(),
               indicators.size());

  return indices;
}

/// @brief Computes equidistribution threshold marking of an indicator.
///
/// Returns the indices \f$i\f$ of the indicators $eta_i$ that satisfy the
/// equidistribution threshold: \f$\eta_i > \theta \frac{||\eta||}{\sqrt{N}} \f$
/// where \f$ N \f$ is the (global) number of indicators.
///
/// @param[in] comm Communicator over which the global equidistribution
/// threshold is computed.
/// @param[in] indicators Indicators (local) \f$ \eta_i \f$ - usually
/// associated with mesh entity \f$ i \f$.
/// @param[in] theta Parameter, \f$ 0 < \theta < 1 \f$.
/// @return Local indices of marked entities.
template <std::floating_point T>
std::vector<std::int32_t>
mark_equidistribution(MPI_Comm comm, std::span<const T> indicators, T theta)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("Theta needs to fullfill 0 < θ < 1.");

  auto norm = std::inner_product(indicators.begin(), indicators.end(),
                                 indicators.begin(), T{0});

  MPI_Allreduce(MPI_IN_PLACE, &norm, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, comm);

  norm = std::sqrt(norm);

  std::int32_t count = indicators.size();
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, dolfinx::MPI::mpi_t<std::int32_t>,
                MPI_SUM, comm);

  auto indices
      = impl::mark_threshold<T>(indicators, theta * norm / std::sqrt(count));

  spdlog::info("Marking (equidistribution) {} / {} (local) entities.",
               indices.size(), indicators.size());

  return indices;
}

/// @brief Computes equidistribution threshold marking of a squared indicator.
///
/// Returns the indices \f$i\f$ of the squared indicators $eta_i^2$ that satisfy
/// the equidistribution threshold: \f$ \eta_i^2 > \theta^2 \frac{||\eta||^2}{N}
/// \f$ where \f$ N \f$ is the (global) number of indicators.
///
/// @param[in] comm Communicator over which the global equidistribution
/// threshold is computed.
/// @param[in] indicators Input indicators (local) \f$ \eta^2_i \f$ -
/// usually associated with mesh entity \f$ i \f$.
/// @param[in] theta Parameter, \f$ 0 < \theta < 1 \f$.
/// @return Local indices of marked entities.
template <std::floating_point T>
std::vector<std::int32_t>
mark_equidistribution_squared(MPI_Comm comm, std::span<const T> indicators,
                              T theta)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("Theta needs to fullfill 0 < θ < 1.");

  auto norm = std::accumulate(indicators.begin(), indicators.end(), T{0});

  MPI_Allreduce(MPI_IN_PLACE, &norm, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, comm);

  std::int32_t count = indicators.size();
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, dolfinx::MPI::mpi_t<std::int32_t>,
                MPI_SUM, comm);

  auto indices
      = impl::mark_threshold<T>(indicators, std::pow(theta, 2) * norm / count);

  spdlog::info("Marking (equidistribution) {} of {} local entities.",
               indices.size(), indicators.size());

  return indices;
}

} // namespace dolfinx::refinement
