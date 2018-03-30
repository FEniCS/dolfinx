// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <map>
#include <vector>

namespace dolfin
{
namespace mesh
{

/// This class stores mesh partitioning data, i.e. the destination
/// process of each cell. For ghosted meshes, some cells may have
/// multiple destinations so the data is stored in CSR format with an
/// offset vector

class PartitionData
{
public:
  /// Build CSR list of processes for each cell from legacy data
  /// @param cell_partition
  ///    Owning process of each cell
  /// @param ghost_procs
  ///    Map of cell_index to vector of sharing processes for those cells
  ///    that have multiple owners
  PartitionData(const std::vector<int>& cell_partition,
                const std::map<std::int64_t, std::vector<int>>& ghost_procs)
      : _offset(1)
  {
    for (std::uint32_t i = 0; i != cell_partition.size(); ++i)
    {
      const auto it = ghost_procs.find(i);
      if (it == ghost_procs.end())
        _dest_processes.push_back(cell_partition[i]);
      else
      {
        _dest_processes.insert(_dest_processes.end(), it->second.begin(),
                               it->second.end());
      }
      _offset.push_back(_dest_processes.size());
    }
  }

  /// Copy constructor
  PartitionData(const PartitionData&) = default;

  /// Move constructor
  PartitionData(PartitionData&&) = default;

  /// Destructor
  ~PartitionData() = default;

  /// Copy Assignment
  PartitionData& operator=(const PartitionData&) = default;

  /// Move Assignment
  PartitionData& operator=(PartitionData&&) = default;

  /// The number of sharing processes of a given cell
  /// @return std::uint32_t
  std::uint32_t num_procs(std::uint32_t i) const
  {
    return _offset[i + 1] - _offset[i];
  }

  /// Pointer to sharing processes of a given cell
  /// @return std::uint32_t*
  const std::uint32_t* procs(std::uint32_t i) const
  {
    return _dest_processes.data() + _offset[i];
  }

  /// Number of cells which this partition is defined for
  /// on this process
  /// @return std::uint32_t
  std::uint32_t size() const { return _offset.size() - 1; }

  /// Return the total number of ghosts cells in this partition on this
  /// process. Useful for testing
  /// @return int
  int num_ghosts() const { return _offset.size() - _dest_processes.size() - 1; }

private:
  // Contiguous list of processes, indexed with offset, below
  std::vector<std::uint32_t> _dest_processes;

  // Index offset for each cell into list
  std::vector<std::uint32_t> _offset;
};
} // namespace mesh
} // namespace dolfin
