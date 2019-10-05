// Copyright (C) 2018 Chris N Richardson
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <map>
#include <utility>
#include <vector>

namespace dolfin
{
namespace mesh
{

/// This class stores mesh partitioning data, i.e. the destination
/// process of each cell. For ghosted meshes, some cells may have
/// multiple destinations so the data is stored in CSR format with an
/// offset vector.

class PartitionData
{
public:
  /// Build CSR list of processes for each cell from legacy data
  /// @param[in] cell_partition Owning process of each cell
  /// @param[in] ghost_procs Map of cell_index to vector of sharing
  ///                        processes for those cells that have
  ///                        multiple owners
  PartitionData(const std::vector<int>& cell_partition,
                const std::map<std::int64_t, std::vector<int>>& ghost_procs);

  /// Build CSR list of processes for each cell from legacy data
  /// @param[in] data Input data
  PartitionData(
      const std::pair<std::vector<int>,
                      std::map<std::int64_t, std::vector<int>>>& data);

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
  std::int32_t num_procs(std::int32_t i) const;

  /// Pointer to sharing processes of a given cell
  /// @param[in] i Cell index
  const std::int32_t* procs(std::int32_t i) const;

  /// Number of cells which this partition is defined for on this
  /// process
  std::int32_t size() const;

  /// Return the total number of ghost cells in this partition on this
  /// process. Useful for testing
  std::int32_t num_ghosts() const;

private:
  // Contiguous list of processes, indexed with offset, below
  std::vector<std::int32_t> _dest_processes;

  // Index offset for each cell into list
  std::vector<std::int32_t> _offset;
};
} // namespace mesh
} // namespace dolfin
