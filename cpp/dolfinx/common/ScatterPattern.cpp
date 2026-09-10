// Copyright (C) 2022-2026 Igor Baratta, Garth N. Wells and Jack S. Hale
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ScatterPattern.h"
#include "IndexMap.h"
#include "MPI.h"
#include "sort.h"
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <iterator>
#include <mpi.h>
#include <numeric>
#include <span>
#include <vector>

using namespace dolfinx;

//-----------------------------------------------------------------------------
common::ScatterPattern::ScatterPattern(const IndexMap& map)
    : _sizes_remote(map.src().size(), 0), _displs_remote(map.src().size() + 1),
      _sizes_local(map.dest().size()), _displs_local(map.dest().size() + 1)
{
  if (dolfinx::MPI::size(map.comm()) == 1)
    return;

  int ierr;

  const std::span<const int> src = map.src();
  const std::span<const int> dest = map.dest();

  // Check that src and dest ranks are unique and sorted
  assert(std::ranges::is_sorted(src));
  assert(std::ranges::is_sorted(dest));

  // Create communicators with directed edges:
  // (0) owner -> ghost,
  // (1) ghost -> owner
  MPI_Comm comm0;
  ierr = MPI_Dist_graph_create_adjacent(
      map.comm(), src.size(), src.data(), MPI_UNWEIGHTED, dest.size(),
      dest.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm0);
  _comm0 = dolfinx::MPI::Comm(comm0, false);
  dolfinx::MPI::check_error(map.comm(), ierr);

  MPI_Comm comm1;
  ierr = MPI_Dist_graph_create_adjacent(
      map.comm(), dest.size(), dest.data(), MPI_UNWEIGHTED, src.size(),
      src.data(), MPI_UNWEIGHTED, MPI_INFO_NULL, false, &comm1);
  _comm1 = dolfinx::MPI::Comm(comm1, false);
  dolfinx::MPI::check_error(map.comm(), ierr);

  // Build permutation array that sorts ghost indices by owning rank
  std::span owners = map.owners();
  _perm.resize(owners.size());
  std::iota(_perm.begin(), _perm.end(), 0);
  dolfinx::radix_sort(_perm, [&owners](std::int32_t i) { return owners[i]; });

  // Sort (i) ghost indices and (ii) ghost index owners by rank (using
  // the permutation array)
  std::span ghosts = map.ghosts();
  std::vector<int> owners_sorted(owners.size());
  std::vector<std::int64_t> ghosts_sorted(owners.size());
  std::ranges::transform(_perm, owners_sorted.begin(),
                         [&owners](std::int32_t i) { return owners[i]; });
  std::ranges::transform(_perm, ghosts_sorted.begin(),
                         [&ghosts](std::int32_t i) { return ghosts[i]; });

  // For data associated with ghost indices, packed by owning
  // (neighbourhood) rank, compute sizes and displacements. I.e., when
  // sending ghost index data from this rank to the owning ranks,
  // disp[i] is the first entry in the buffer sent to neighbourhood rank
  // i, and disp[i + 1] - disp[i] is the number of values sent to rank i.
  assert(_sizes_remote.size() == src.size());
  assert(_displs_remote.size() == src.size() + 1);
  auto begin = owners_sorted.begin();
  for (std::size_t i = 0; i < src.size(); i++)
  {
    auto upper = std::upper_bound(begin, owners_sorted.end(), src[i]);
    std::size_t num_ind = std::ranges::distance(begin, upper);
    _displs_remote[i + 1] = _displs_remote[i] + num_ind;
    _sizes_remote[i] = num_ind;
    begin = upper;
  }

  // For data associated with owned indices that are ghosted by other
  // ranks, compute the size and displacement arrays. When sending data
  // associated with ghost indices to the owner, these size and
  // displacement arrays are for the receive buffer.

  // Compute sizes and displacements of local data (how many local
  // elements to be sent/received grouped by neighbours)
  assert(_sizes_local.size() == dest.size());
  assert(_displs_local.size() == dest.size() + 1);
  // Allocate so that data() is not null when a rank has no neighbours
  _sizes_remote.reserve(1);
  _sizes_local.reserve(1);
  ierr = MPI_Neighbor_alltoall(_sizes_remote.data(), 1, MPI_INT,
                               _sizes_local.data(), 1, MPI_INT, _comm1.comm());
  dolfinx::MPI::check_error(_comm1.comm(), ierr);

  std::partial_sum(_sizes_local.begin(), _sizes_local.end(),
                   std::next(_displs_local.begin()));

  assert(static_cast<int>(ghosts_sorted.size()) == _displs_remote.back());

  // Send ghost global indices to owning rank, and receive owned indices
  // that are ghosts on other ranks
  std::vector<std::int64_t> recv_buffer(_displs_local.back(), 0);
  ierr = MPI_Neighbor_alltoallv(
      ghosts_sorted.data(), _sizes_remote.data(), _displs_remote.data(),
      MPI_INT64_T, recv_buffer.data(), _sizes_local.data(),
      _displs_local.data(), MPI_INT64_T, _comm1.comm());
  dolfinx::MPI::check_error(_comm1.comm(), ierr);

  const std::array<std::int64_t, 2> range = map.local_range();
#ifndef NDEBUG
  // Check that all received indices are within the owned range
  std::ranges::for_each(recv_buffer, [&range](std::int64_t idx)
                        { assert(idx >= range[0] and idx < range[1]); });
#endif

  // Convert the received indices from global to local numbering
  _local_inds.resize(recv_buffer.size());
  std::ranges::transform(recv_buffer, _local_inds.begin(),
                         [&range](std::int64_t idx)
                         { return static_cast<std::int32_t>(idx - range[0]); });
}
//-----------------------------------------------------------------------------
