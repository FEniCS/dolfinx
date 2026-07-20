// Copyright (C) 2006-2024 Anders Logg and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <cstdint>
#include <dolfinx/graph/AdjacencyList.h>
#include <functional>
#include <memory>
#include <tuple>
#include <vector>

namespace dolfinx::common
{
class IndexMap;
}

namespace dolfinx::mesh
{
class Topology;
enum class CellType : std::int8_t;

/// @brief Callback invoked from within a worker thread before it starts
/// its share of work, used to apply thread placement (e.g. pinning to a
/// specific core, typically within the NUMA domain the calling MPI rank
/// is bound to).
///
/// Called with `(thread_index, num_threads)`, where `thread_index` is
/// in `[0, num_threads)`. Invoked on the worker thread itself (not the
/// spawning thread), so it may safely call platform affinity APIs (e.g.
/// `pthread_setaffinity_np`) on the calling thread's native handle.
using AffinityPolicy = std::function<void(int, int)>;

/// @brief Compute mesh entities of given topological dimension by
/// computing cell-to-entity `(tdim, i) -> `(dim, entity_type)` and
/// entity-to-vertex connectivity `(dim, entity_type) -> `(0, 0)`
/// connectivity.
///
/// Computed entities are oriented such that their local (to the
/// process) orientation agrees with their global orientation
///
/// @param[in] topology Mesh topology.
/// @param[in] dim Dimension of the entities to create.
/// @param[in] entity_type Entity type in dimension `dim` to create.
/// Entity type must be in the list returned by Topology::entity_types.
/// @param[in] num_threads Number of threads to use for entity creation.
/// Must be >= 1.
/// @param[in] affinity_policy Optional callback applied by each worker
/// thread before it begins its share of work, e.g. to pin the thread to
/// a core. If empty (the default), no affinity is set and threads run
/// with whatever affinity they inherit.
///
/// @return Tuple of (cell->entity connectivity, entity->vertex
/// connectivity, index map for created entities, list of interprocess
/// entities). Interprocess entities lie on the "true" boundary between
/// owned cells of each process. If entities of type `entity_type`
/// already exists, then {nullptr, nullptr, nullptr, std::vector()} is
/// returned.
std::tuple<std::vector<std::shared_ptr<graph::AdjacencyList<std::int32_t>>>,
           std::shared_ptr<graph::AdjacencyList<std::int32_t>>,
           std::shared_ptr<common::IndexMap>, std::vector<std::int32_t>>
compute_entities(const Topology& topology, int dim, CellType entity_type,
                 int num_threads = 1, AffinityPolicy affinity_policy = nullptr);

/// @brief Compute connectivity (d0 -> d1) for given pair of entity
/// types, given by topological dimension and index, as found in
/// `Topology::entity_types()`
/// @param[in] topology The topology
/// @param[in] d0 Dimension and index of the entities, `(dim0, i)`.
/// @param[in] d1 Dimension and index of the incident entities, `(dim1,
/// j)`.
/// @returns The connectivities [(d0 -> d1), (d1 -> d0)] if they are
/// computed. If (d0, d1) already exists then a nullptr is returned. If
/// (d0, d1) is computed and the computation of (d1, d0) was required as
/// part of computing (d0, d1), the (d1, d0) is returned as the second
/// entry. The second entry is otherwise nullptr.
std::array<std::shared_ptr<graph::AdjacencyList<std::int32_t>>, 2>
compute_connectivity(const Topology& topology, std::array<int, 2> d0,
                     std::array<int, 2> d1);

} // namespace dolfinx::mesh
