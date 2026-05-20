// Copyright (C) 2026 Paul T. Kühner, Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <limits>
#include <mpi.h>
#include <numeric>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <vector>

#include "dolfinx/common/MPI.h"

namespace dolfinx::refinement
{

namespace impl
{

/// @brief Computes local threshold marking of indicators.
///
/// Helper for other marking routines.
///
/// Returns the indices \f$ i \f$ of the indicators \f$ \eta_i \f$ that satisfy
/// the threshold: \f$ \eta_i > \text{threshold} \f$.
///
/// @param[in] indicators Indicators (local) \f$ \eta_i \f$ -
/// usually an error indicator associated with mesh entity \f$ i \f$.
/// @param[in] threshold Threshold value; indicators greater than this are
/// marked.
/// @return Local indices of marked entities.
template <std::floating_point T>
std::vector<std::int32_t> mark_threshold(std::span<const T> indicators,
                                         T threshold)
{
  std::vector<std::int32_t> indices;
  indices.reserve(indicators.size());
  for (std::int32_t i = 0; i < static_cast<std::int32_t>(indicators.size());
       ++i)
  {
    if (indicators[i] > threshold)
      indices.push_back(i);
  }

  return indices;
}

} // namespace impl

/// @brief Computes maximum-based marking of indicators.
///
/// Returns the indices \f$ i \f$ of the indicators \f$ \eta_i \f$ that satisfy
/// the maximum threshold: \f$ \eta_i > \theta \max_j \eta_j \f$.
///
/// @param[in] comm Communicator to compute the maximum over.
/// @param[in] indicators Indicators (local) \f$ \eta_i \f$ -
/// usually an error indicator associated with mesh entity \f$ i \f$.
/// @param[in] theta Parameter, \f$ 0 < \theta < 1 \f$.
/// @return Local indices of marked entities.
template <std::floating_point T>
std::vector<std::int32_t> mark_maximum(MPI_Comm comm,
                                       std::span<const T> indicators, T theta)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("theta must fulfill 0 < theta < 1.");

  T max = indicators.empty() ? std::numeric_limits<T>::lowest()
                             : std::ranges::max(indicators);
  MPI_Allreduce(MPI_IN_PLACE, &max, 1, dolfinx::MPI::mpi_t<T>, MPI_MAX, comm);

  std::vector<std::int32_t> indices
      = impl::mark_threshold<T>(indicators, theta * max);

  spdlog::info("Marking (maximum) {} / {} (local) entities.", indices.size(),
               indicators.size());

  return indices;
}

/// @brief Computes equidistribution threshold marking of indicators.
///
/// Returns the indices \f$ i \f$ of the indicators \f$ \eta_i \f$ that satisfy
/// the equidistribution threshold: \f$\eta_i > \theta
/// \frac{||\eta||}{\sqrt{N}} \f$ where \f$ N \f$ is the (global) number of
/// indicators.
///
/// @param[in] comm Communicator over which the global equidistribution
/// threshold is computed.
/// @param[in] indicators Indicators (local) \f$ \eta_i \f$ - usually
/// associated with mesh entity \f$ i \f$.
/// @param[in] theta Parameter, \f$ 0 < \theta < 1 \f$.
/// @return Local indices of indicators that satisfy the threshold.
template <std::floating_point T>
std::vector<std::int32_t>
mark_equidistribution(MPI_Comm comm, std::span<const T> indicators, T theta)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("theta must fulfill 0 < theta < 1.");

  T norm = std::inner_product(indicators.begin(), indicators.end(),
                              indicators.begin(), T{0});

  MPI_Allreduce(MPI_IN_PLACE, &norm, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, comm);

  T sqrt_norm = std::sqrt(norm);

  // int64_t gives headroom for global sum across ranks.
  std::int64_t count = indicators.size();
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, dolfinx::MPI::mpi_t<std::int64_t>,
                MPI_SUM, comm);

  std::vector<std::int32_t> indices = impl::mark_threshold<T>(
      indicators, theta * sqrt_norm / std::sqrt(static_cast<T>(count)));

  spdlog::info("Marking (equidistribution) {} / {} (local) entities.",
               indices.size(), indicators.size());

  return indices;
}

/// @brief Computes equidistribution threshold marking of a squared indicator.
///
/// Returns the indices \f$i\f$ of the squared indicators \f$ \eta_i^2 \f$ that
/// satisfy the equidistribution threshold: \f$ \eta_i^2 > \theta^2
/// \frac{||\eta||^2}{N} \f$ where \f$ N \f$ is the (global) number of
/// indicators.
///
/// @param[in] comm Communicator over which the global equidistribution
/// threshold is computed.
/// @param[in] squared_indicators Input squared indicators (local) \f$ \eta^2_i \f$ -
/// usually associated with mesh entity \f$ i \f$.
/// @param[in] theta Parameter, \f$ 0 < \theta < 1 \f$.
/// @return Local indices of squared indicators that satisfy the threshold.
template <std::floating_point T>
std::vector<std::int32_t>
mark_equidistribution_squared(MPI_Comm comm,
                              std::span<const T> squared_indicators, T theta)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("theta must fulfill 0 < theta < 1.");

  T norm = std::accumulate(squared_indicators.begin(),
                           squared_indicators.end(), T{0});

  MPI_Allreduce(MPI_IN_PLACE, &norm, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, comm);

  // int64_t gives headroom for global sum across ranks.
  std::int64_t count = squared_indicators.size();
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, dolfinx::MPI::mpi_t<std::int64_t>,
                MPI_SUM, comm);

  std::vector<std::int32_t> indices = impl::mark_threshold<T>(
      squared_indicators, std::pow(theta, 2) * norm / static_cast<T>(count));

  spdlog::info("Marking (equidistribution) {} of {} local entities.",
               indices.size(), squared_indicators.size());

  return indices;
}

} // namespace dolfinx::refinement
