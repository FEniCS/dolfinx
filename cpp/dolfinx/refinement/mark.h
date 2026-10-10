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
#include <numeric>
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
/// Computes the maximum \f$ \max_j v_j \f$ of @p values \f$ v \f$ over the
/// locally owned entries on every rank of `index_map.comm()`, and returns the
/// local indices \f$ i \f$ satisfying \f$ v_i > \theta \max_j v_j \f$. This is
/// commonly referred to as 'maximum marking' in the adaptive finite element
/// literature.
///
/// @pre @p values has size `index_map.size_local() + index_map.num_ghosts()`.
/// @pre Ghost entries of @p values are up to date, i.e. `scatter_forward` has
/// been called since the owned entries were last modified.
///
/// @note \f$ \theta = 1 \f$ marks nothing, since no entry can strictly exceed
/// the true maximum. \f$ \theta = 0 \f$ is rejected, since the threshold would
/// be 0 and the criterion would degenerate to marking every entry with a
/// positive value.
///
/// @param[in] values Values, often with each entry associated with a mesh
///   entity, e.g. an error indicator.
/// @param[in] index_map Index map describing the parallel layout of @p
///   values.
/// @param[in] theta Cut-off parameter, \f$ 0 < \theta \leq 1 \f$.
/// @return Local indices, ascending and including ghosts, of @p values
/// satisfying \f$ v_i > \theta \max_j v_j \f$.
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

/// @brief Return local indices of a set of values whose entry exceeds a
/// fraction of the mean square (MS).
///
/// Computes the mean \f$ \frac{1}{N} \sum_j v_j \f$ of @p values \f$ v \f$
/// over the locally owned entries on every rank of `index_map.comm()`, where
/// \f$ N \f$ is the global number of entries, and returns the local indices
/// \f$ i \f$ satisfying \f$ v_i > \frac{\theta^2}{N} \sum_j v_j \f$.
///
/// Each entry is expected to hold a squared error indicator, \f$ v_i =
/// \eta_i^2 \f$, so that \f$ \frac{1}{N} \sum_j v_j \f$ is the mean square
/// (MS) \f$ \|\eta\|_2^2 / N \f$ of the indicators and the criterion reads
/// \f$ \eta_i^2 > \theta^2 \|\eta\|_2^2 / N \f$. This is commonly referred to
/// as 'equidistribution marking' in the adaptive finite element literature.
///
/// @pre @p values has size `index_map.size_local() + index_map.num_ghosts()`.
/// @pre Ghost entries of @p values are up to date, i.e. `scatter_forward` has
/// been called since the owned entries were last modified.
///
/// @note \f$ \theta = 0 \f$ is rejected, since the threshold would be 0 and
/// the criterion would degenerate to marking every entry with a positive
/// value.
///
/// @param[in] values Values, often with each entry associated with a mesh
///   entity, e.g. a squared error indicator \f$ \eta_i^2 \f$.
/// @param[in] index_map Index map describing the parallel layout of @p
///   values.
/// @param[in] theta Cut-off parameter, \f$ 0 < \theta \leq 1 \f$.
/// @return Local indices, ascending and including ghosts, of @p values
/// satisfying \f$ v_i > \frac{\theta^2}{N} \sum_j v_j \f$.
template <std::floating_point T>
std::vector<std::int32_t>
mark_equidistribution(std::span<const T> values,
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

  auto owned = values.first(index_map.size_local());
  T local_squared_norm = std::accumulate(owned.begin(), owned.end(), T{0});

  T squared_norm;
  MPI_Allreduce(&local_squared_norm, &squared_norm, 1, dolfinx::MPI::mpi_t<T>,
                MPI_SUM, index_map.comm());

  T threshold
      = theta * theta * squared_norm / static_cast<T>(index_map.size_global());

  auto mark = [threshold](T e) { return e > threshold; };

  std::vector<std::int32_t> indices;
  indices.reserve(std::ranges::count_if(values, mark));
  for (std::int32_t i = 0; i < n; ++i)
  {
    if (mark(values[i]))
      indices.push_back(i);
  }

  spdlog::info(
      "Marking (equidistribution): marked {} of {} local entries (owned + "
      "ghost).",
      indices.size(), n);

  return indices;
}

} // namespace dolfinx::refinement
