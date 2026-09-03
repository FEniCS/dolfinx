// Copyright (C) 2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "AdjacencyList.h"
#include "partition.h"
#include <cstdint>
#include <mpi.h>
#include <optional>
#include <span>
#include <vector>

/// @file sfc.h
/// @brief Point partitioning by position along a space-filling curve.

namespace dolfinx::graph
{
/// @brief Partition points into `nparts` groups using a Morton ('Z-order')
/// space-filling curve.
///
/// Points are ordered by their Morton keys in the global bounding box and
/// divided into groups with approximately equal numbers of points or total
/// weight. For improved spatial locality, see ::partition_sfc_hilbert.
///
/// @note Collective.
///
/// @note There is no graph, so this cannot ghost: it always assigns
/// exactly one destination per point.
///
/// @param[in] comm MPI communicator that the points are distributed
/// across.
/// @param[in] nparts Number of partitions to divide the points into.
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point. Must be
/// 1, 2 or 3.
/// @param[in] weights Point weights, one entry per row of `x`.
/// Partitions aim for equal sums of weight along the curve rather than
/// equal counts. If `std::nullopt`, points are treated as having equal
/// weight.
/// @return Destination rank for each point, one entry per row of `x`.
std::vector<int> partition_sfc_morton(
    MPI_Comm comm, int nparts, std::span<const double> x, int gdim,
    std::optional<std::span<const std::int32_t>> weights = std::nullopt);

/// @brief Partition points into `nparts` groups using a Hilbert
/// space-filling curve.
///
/// As ::partition_sfc_morton, but uses Hilbert keys, which generally preserve
/// spatial locality better than Morton keys.
std::vector<int> partition_sfc_hilbert(
    MPI_Comm comm, int nparts, std::span<const double> x, int gdim,
    std::optional<std::span<const std::int32_t>> weights = std::nullopt);

/// @brief Reorder locally supplied points using a Morton ('Z-order')
/// space-filling curve.
///
/// The bounding box is computed from `x`. The returned ordering is local and
/// does not provide a distributed ordering.
///
/// @param[in] x Point coordinates, row-major with `gdim` columns.
/// @param[in] gdim Number of coordinate components per point. Must be 1, 2 or
/// 3.
/// @return Reordering array `map`, where `map[i]` is the new index of point
/// `i`.
std::vector<std::int32_t> reorder_sfc_morton(std::span<const double> x,
                                             int gdim);

/// @brief Reorder locally supplied points using a Hilbert space-filling
/// curve.
///
/// As ::reorder_sfc_morton, but uses Hilbert keys.
std::vector<std::int32_t> reorder_sfc_hilbert(std::span<const double> x,
                                              int gdim);
} // namespace dolfinx::graph
