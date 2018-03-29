// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

namespace dolfin
{
namespace mesh
{

/// This class stores mesh partitioning data, i.e. the destination process
/// of each cell. For ghosted meshes, some cells may have multiple destinations
/// so the data is stored in CSR format with an offset vector

class MeshPartition
{
public:
  /// Build CSR list of processes for each cell from legacy data
  /// @param cell_partition
  ///    Owning process of each cell
  /// @param ghost_procs
  ///    Map of cell_index to vector of sharing processes for those cells
  ///    that have multiple owners
  MeshPartition(const std::vector<int>& cell_partition,
                const std::map<std::int64_t, std::vector<int>>& ghost_procs)
  {
    offset = {0};

    for (std::uint32_t i = 0; i != cell_partition.size(); ++i)
    {
      const auto it = ghost_procs.find(i);
      if (it == ghost_procs.end())
        dest_processes.push_back(cell_partition[i]);
      else
        dest_processes.insert(dest_processes.end(), it->second.begin(),
                              it->second.end());
      offset.push_back(dest_processes.size());
    }
  }

  /// Copy constructor
  MeshPartition(const MeshPartition&) = default;

  /// Move constructor
  MeshPartition(MeshPartition&&) = default;

  /// Destructor
  ~MeshPartition() = default;

  /// Copy Assignment
  MeshPartition& operator=(const MeshPartition&) = default;

  /// Move Assignment
  MeshPartition& operator=(MeshPartition&&) = default;

  /// The number of sharing processes of a given cell
  /// @return std::uint32_t
  ///
  std::uint32_t num_procs(std::uint32_t i) const
  {
    return offset[i + 1] - offset[i];
  }

  /// Pointer to sharing processes of a given cell
  /// @return std::uint32_t*
  ///
  const std::uint32_t* procs(std::uint32_t i) const
  {
    return dest_processes.data() + offset[i];
  }

  /// Number of cells which this partition is defined for
  /// on this process
  /// @return std::uint32_t
  ///
  std::uint32_t size() const { return offset.size() - 1; }

  /// Return the total number of ghosts cells in this partition on this
  /// process. Useful for testing
  /// @return int
  ///
  int num_ghosts() const { return offset.size() - dest_processes.size() - 1; }

private:
  // Contiguous list of processes, indexed with offset, below
  std::vector<std::uint32_t> dest_processes;

  // Index offset for each cell into list
  std::vector<std::uint32_t> offset;
};
} // namespace mesh
} // namespace dolfin
