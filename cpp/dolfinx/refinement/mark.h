// Copyright (C) 2026 Paul T. Kühner and Jack S. Hale
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <dolfinx/common/IndexMap.h>
#include <dolfinx/common/MPI.h>
#include <dolfinx/la/Vector.h>
#include <format>
#include <limits>
#include <mpi.h>
#include <span>
#include <spdlog/spdlog.h>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dolfinx::refinement
{

/// @brief Return local indices of a set of values whose entry exceeds a
/// fraction of the global maximum value.
///
/// Computes the maximum `max` of @p values over the locally owned entries on
/// every rank of `index_map.comm()`, and returns the local indices `i`,
/// satisfying `values[i] > θ max`. This is commonly referred to as 'maximum
/// marking' in the adaptive finite element literature.
///
/// @pre @p values has size `index_map.size_local() + index_map.num_ghosts()`.
/// @pre Ghost entries of @p values are up to date, i.e. `scatter_forward` has
/// been called since the owned entries were last modified.
///
/// @note θ = 1 marks nothing, since no entry can strictly exceed the true
/// maximum. θ = 0 is rejected, since `threshold` would be 0 and the
/// criterion would degenerate to marking every entry with a positive
/// value.
///
///
/// @param[in] values Values, often with each entry associated with a mesh
///   entity, e.g. an error indicator.
/// @param[in] index_map Index map describing the parallel layout of @p
///   values.
/// @param[in] theta Cut-off parameter, 0 < θ ≤ 1.
/// @return Local indices, ascending and including ghosts, of `values`
/// that satisfy `values[i] > θ max`.
template <std::floating_point T>
std::vector<std::int32_t> mark_maximum(std::span<const T> values,
                                       const common::IndexMap& index_map,
                                       std::type_identity_t<T> theta)
{
  if ((theta <= 0) or (theta > 1))
  {
    throw std::invalid_argument(
        std::format("theta must satisfy 0 < theta <= 1, got {}.", theta));
  }

  std::int32_t n = values.size();
  std::int32_t size = index_map.size_local() + index_map.num_ghosts();
  if (n != size)
  {
    throw std::invalid_argument(
        std::format("values must have size index_map.size_local() + "
                    "index_map.num_ghosts() = {}, got {}.",
                    size, n));
  }

  T local_max = index_map.size_local() == 0
                    ? std::numeric_limits<T>::lowest()
                    : std::ranges::max(values.first(index_map.size_local()));

  T max = 0;
  MPI_Allreduce(&local_max, &max, 1, dolfinx::MPI::mpi_t<T>, MPI_MAX,
                index_map.comm());

  T threshold = theta * max;

  auto mark = [threshold](T e) { return e > threshold; };

  std::vector<std::int32_t> indices;
  indices.reserve(std::ranges::count_if(values, mark));
  for (std::int32_t i = 0; i < n; ++i)
  {
    if (mark(values[i]))
      indices.push_back(i);
  }

  spdlog::info("Marking (maximum): marked {} of {} local entries (owned + "
               "ghost).",
               indices.size(), n);

  return indices;
}

/// @brief Computes equidistribution threshold marking of indicators.
///Expand commentComment on line R94Resolved
/// Returns the indices \f$ i \f$ of the indicators \f$ \eta_i \f$ that satisfy
/// the equidistribution threshold: \f$\eta_i > \theta
/// \frac{||\eta||}{\sqrt{N}} \f$ where \f$ N \f$ is the (global) number of
/// indicators.
///
/// @warning `indicators` must contain owned entities only. Ghost values cause
/// double-counting in the global sum reductions.
///
/// @param[in] comm Communicator over which the global equidistribution
/// threshold is computed.
/// @param[in] indicators Indicators for owned local entities \f$ \eta_i \f$
/// - usually associated with mesh entity \f$ i \f$.
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
/// @warning `squared_indicators` must contain owned entities only. Ghost
/// values cause double-counting in the global sum reductions.
///
/// @param[in] comm Communicator over which the global equidistribution
/// threshold is computed.
/// @param[in] squared_indicators Input squared indicators for owned local
/// entities \f$ \eta^2_i \f$ - usually associated with mesh entity \f$ i \f$.
/// @param[in] theta Parameter, \f$ 0 < \theta < 1 \f$.
/// @return Local indices of squared indicators that satisfy the threshold.
template <std::floating_point T>
std::vector<std::int32_t>
mark_equidistribution_squared(MPI_Comm comm,
                              std::span<const T> squared_indicators, T theta)
{
  if ((theta <= 0) || (theta >= 1))
    throw std::invalid_argument("theta must fulfill 0 < theta < 1.");

  T norm = std::accumulate(squared_indicators.begin(), squared_indicators.end(),
                           T{0});

  MPI_Allreduce(MPI_IN_PLACE, &norm, 1, dolfinx::MPI::mpi_t<T>, MPI_SUM, comm);

  // int64_t gives headroom for global sum across ranks.
  std::int64_t count = squared_indicators.size();
  MPI_Allreduce(MPI_IN_PLACE, &count, 1, dolfinx::MPI::mpi_t<std::int64_t>,
                MPI_SUM, comm);

  std::vector<std::int32_t> indices = impl::mark_threshold<T>(
      squared_indicators, theta * theta * norm / static_cast<T>(count));

  spdlog::info("Marking (equidistribution) {} of {} local entities.",
               indices.size(), squared_indicators.size());

  return indices;
}

} // namespace dolfinx::refinement
