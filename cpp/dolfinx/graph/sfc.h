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
#include <span>
#include <vector>

/// @file sfc.h
/// @brief Point partitioning by position along a space-filling curve.

namespace dolfinx::graph
{
/// @brief Partition points into `nparts` groups of (approximately) equal
/// size using a Morton ('Z-order') space-filling curve.
///
/// Points are ordered by the Morton key of their position in the global
/// bounding box, and the resulting order is cut into `nparts` equal
/// pieces. Splitters are selected from a gathered sample of the keys, so
/// the cost is linear in the number of local points plus one all-gather
/// of the sample.
///
/// Compared to a graph partitioner, this is much cheaper (no graph is
/// required and the cost is nearly independent of the number of ranks)
/// and gives a near-perfect load balance, at the cost of a larger number
/// of cut edges (typically tens of percent for a mesh dual graph).
///
/// A Morton curve jumps a long way in space each time a high bit of the
/// key changes, so consecutive points on the curve are not always close
/// together. ::partition_sfc_hilbert avoids this.
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
/// @return Destination rank for each point, one entry per row of `x`.
std::vector<int> partition_sfc_morton(MPI_Comm comm, int nparts,
                                      std::span<const double> x, int gdim);

/// @brief Partition points into `nparts` groups of (approximately) equal
/// size using a Hilbert space-filling curve.
///
/// As ::partition_sfc_morton, but points are ordered along a Hilbert
/// curve. Successive points on a Hilbert curve are always neighbours in
/// space, which a Morton curve does not guarantee, so the resulting
/// partitions are more compact and cut fewer edges. Computing the curve
/// index is more expensive than a Morton key, but in both cases the cost
/// is dominated by the sampling and the search for each point's part.
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
/// @return Destination rank for each point, one entry per row of `x`.
std::vector<int> partition_sfc_hilbert(MPI_Comm comm, int nparts,
                                       std::span<const double> x, int gdim);

/// Space-filling curve partitioner
namespace sfc
{
/// @brief Space-filling curves that nodes can be ordered along.
enum class curve : std::uint8_t
{
  /// Morton ('Z-order') curve, see ::partition_sfc_morton
  morton,

  /// Hilbert curve, see ::partition_sfc_hilbert
  hilbert
};

/// @brief Create a geometric partitioning function that orders nodes
/// along a space-filling curve.
///
/// The partition is computed from the node coordinates alone.
///
/// @note The default is curve::hilbert. For the dual graph of a
/// tetrahedral mesh on 20 ranks, it was measured to cut 10% fewer edges
/// than curve::morton (457772 against 508750 for 12.6M cells), for a
/// similar cost.
///
/// @note ::geom_partition_fn has no graph, so the returned function
/// has no `ghosting` parameter and is never asked to ghost.
///
/// @param[in] curve Space-filling curve to order the nodes along.
/// @return A geometric graph partitioning function. It requires `x`.
graph::geom_partition_fn partitioner(sfc::curve curve = sfc::curve::hilbert);
} // namespace sfc

} // namespace dolfinx::graph
