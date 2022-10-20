// Copyright (C) 2022 Igor Baratta and Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Scatterer.h"
#include "IndexMap.h"
#include "sort.h"
#include <algorithm>
#include <mpi.h>
#include <numeric>

using namespace dolfinx;
using namespace dolfinx::common;

//-----------------------------------------------------------------------------
Scatterer::Scatterer(const IndexMap& map, int bs)
    : _bs(bs), _comm0(MPI_COMM_NULL), _comm1(MPI_COMM_NULL)
{
  if (map.overlapped())
  {
    // Get source (owner of ghosts) and destination (processes that
    // ghost an owned index) ranks
    const std::vector<int>& src_ranks = map.src();
    const std::vector<int>& dest_ranks = map.dest();

    // Check that src and dest ranks are unique and sorted
    assert(std::is_sorted(src_ranks.begin(), src_ranks.end()));
    assert(std::is_sorted(dest_ranks.begin(), dest_ranks.end()));

    // Create communicators with directed edges:
    // (0) owner -> ghost,
    // (1) ghost -> owner
    MPI_Comm comm0;
    int err = MPI_Dist_graph_create_adjacent(
        map.comm(), src_ranks.size(), src_ranks.data(), MPI_UNWEIGHTED,
        dest_ranks.size(), dest_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
        false, &comm0);
    _comm0 = dolfinx::MPI::Comm(comm0, false);
    dolfinx::MPI::check_error(map.comm(), err);

    MPI_Comm comm1;
    err = MPI_Dist_graph_create_adjacent(
        map.comm(), dest_ranks.size(), dest_ranks.data(), MPI_UNWEIGHTED,
        src_ranks.size(), src_ranks.data(), MPI_UNWEIGHTED, MPI_INFO_NULL,
        false, &comm1);
    _comm1 = dolfinx::MPI::Comm(comm1, false);
    dolfinx::MPI::check_error(map.comm(), err);

    // Build permutation array that sorts ghost indices by owning rank
    const std::vector<int>& owners = map.owners();
    std::vector<std::int32_t> perm(owners.size());
    std::iota(perm.begin(), perm.end(), 0);
    dolfinx::argsort_radix<std::int32_t>(owners, perm);

    // Sort (i) ghost indices and (ii) ghost index owners by rank
    // (using perm array)
    const std::vector<std::int64_t>& ghosts = map.ghosts();
    std::vector<int> owners_sorted(owners.size());
    std::vector<std::int64_t> ghosts_sorted(owners.size());
    std::transform(perm.begin(), perm.end(), owners_sorted.begin(),
                   [&owners](auto idx) { return owners[idx]; });
    std::transform(perm.begin(), perm.end(), ghosts_sorted.begin(),
                   [&ghosts](auto idx) { return ghosts[idx]; });

    // For data associated with ghost indices, packed by owning
    // (neighbourhood) rank, compute sizes and displacements. I.e.,
    // when sending ghost index data from this rank to the owning
    // ranks, disp[i] is the first entry in the buffer sent to
    // neighbourhood rank i, and disp[i + 1] - disp[i] is the number
    // of values sent to rank i.
    _sizes_remote.resize(src_ranks.size(), 0);
    _displs_remote.resize(src_ranks.size() + 1, 0);
    std::vector<std::int32_t>::iterator begin = owners_sorted.begin();
    for (std::size_t i = 0; i < src_ranks.size(); i++)
    {
      auto upper = std::upper_bound(begin, owners_sorted.end(), src_ranks[i]);
      int num_ind = std::distance(begin, upper);
      _displs_remote[i + 1] = _displs_remote[i] + num_ind;
      _sizes_remote[i] = num_ind;
      begin = upper;
    }

    // For data associated with owned indices that are ghosted by
    // other ranks, compute the size and displacement arrays. When
    // sending data associated with ghost indices to the owner, these
    // size and displacement arrays are for the receive buffer.

    // Compute sizes and displacements of local data (how many local
    // elements to be sent/received grouped by neighbors)
    _sizes_local.resize(dest_ranks.size());
    _displs_local.resize(_sizes_local.size() + 1);
    _sizes_remote.reserve(1);
    _sizes_local.reserve(1);
    err = MPI_Neighbor_alltoall(_sizes_remote.data(), 1, MPI_INT32_T,
                                _sizes_local.data(), 1, MPI_INT32_T,
                                _comm1.comm());
    dolfinx::MPI::check_error(map.comm(), err);
    std::partial_sum(_sizes_local.begin(), _sizes_local.end(),
                     std::next(_displs_local.begin()));

    assert((std::int32_t)ghosts_sorted.size() == _displs_remote.back());
    assert((std::int32_t)ghosts_sorted.size() == _displs_remote.back());

    // Send ghost global indices to owning rank, and receive owned
    // indices that are ghosts on other ranks
    std::vector<std::int64_t> recv_buffer(_displs_local.back(), 0);
    err = MPI_Neighbor_alltoallv(
        ghosts_sorted.data(), _sizes_remote.data(), _displs_remote.data(),
        MPI_INT64_T, recv_buffer.data(), _sizes_local.data(),
        _displs_local.data(), MPI_INT64_T, _comm1.comm());
    dolfinx::MPI::check_error(map.comm(), err);

    const std::array<std::int64_t, 2> range = map.local_range();
#ifndef NDEBUG
    // Check that all received indice are within the owned range
    std::for_each(recv_buffer.begin(), recv_buffer.end(),
                  [range](auto idx)
                  { assert(idx >= range[0] and idx < range[1]); });
#endif

    // Scale sizes and displacements by block size
    {
      auto rescale = [](auto& x, int bs)
      {
        std::transform(x.begin(), x.end(), x.begin(),
                       [bs](auto e) { return e *= bs; });
      };
      rescale(_sizes_local, bs);
      rescale(_displs_local, bs);
      rescale(_sizes_remote, bs);
      rescale(_displs_remote, bs);
    }

    // Expand local indices using block size and convert it from
    // global to local numbering
    _local_inds.resize(recv_buffer.size() * _bs);
    std::int64_t offset = range[0] * _bs;
    for (std::size_t i = 0; i < recv_buffer.size(); i++)
      for (int j = 0; j < _bs; j++)
        _local_inds[i * _bs + j] = (recv_buffer[i] * _bs + j) - offset;

    // Expand remote indices using block size
    _remote_inds.resize(perm.size() * _bs);
    for (std::size_t i = 0; i < perm.size(); i++)
      for (int j = 0; j < _bs; j++)
        _remote_inds[i * _bs + j] = perm[i] * _bs + j;
  }
}
//-----------------------------------------------------------------------------
void Scatterer::scatter_fwd_end(MPI_Request& request) const
{
  // Return early if there are no incoming or outgoing edges
  if (_sizes_local.empty() and _sizes_remote.empty())
    return;

  // Wait for communication to complete
  int err = MPI_Wait(&request, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(_comm0.comm(), err);
}
//-----------------------------------------------------------------------------
void Scatterer::scatter_rev_end(MPI_Request& request) const
{
  // Return early if there are no incoming or outgoing edges
  if (_sizes_local.empty() and _sizes_remote.empty())
    return;

  // Wait for communication to complete
  int err = MPI_Wait(&request, MPI_STATUS_IGNORE);
  dolfinx::MPI::check_error(_comm0.comm(), err);
}
//-----------------------------------------------------------------------------
std::int32_t Scatterer::local_buffer_size() const noexcept
{
  return _local_inds.size();
}
//-----------------------------------------------------------------------------
std::int32_t Scatterer::remote_buffer_size() const noexcept
{
  return _remote_inds.size();
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& Scatterer::local_indices() const noexcept
{
  return _local_inds;
}
//-----------------------------------------------------------------------------
const std::vector<std::int32_t>& Scatterer::remote_indices() const noexcept
{
  return _remote_inds;
}
//-----------------------------------------------------------------------------
int Scatterer::bs() const noexcept { return _bs; }
//-----------------------------------------------------------------------------
