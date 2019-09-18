// Copyright (C) 2018 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PartitionData.h"

#include <iostream>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
PartitionData::PartitionData(
    const std::vector<int>& cell_partition,
    const std::map<std::int64_t, std::vector<int>>& ghost_procs)
    : _offset(1)

{
  for (std::size_t i = 0; i < cell_partition.size(); ++i)
  {
    auto it = ghost_procs.find(i);
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
//-----------------------------------------------------------------------------
PartitionData::PartitionData(
    const std::pair<std::vector<int>, std::map<std::int64_t, std::vector<int>>>&
        data)
    : PartitionData(data.first, data.second)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
std::int32_t PartitionData::num_procs(std::int32_t i) const
{
  return _offset[i + 1] - _offset[i];
}
//-----------------------------------------------------------------------------
const std::int32_t* PartitionData::procs(std::int32_t i) const
{
  return _dest_processes.data() + _offset[i];
}
//-----------------------------------------------------------------------------
std::int32_t PartitionData::size() const { return _offset.size() - 1; }
//-----------------------------------------------------------------------------
int PartitionData::num_ghosts() const
{
  return _offset.size() - _dest_processes.size() - 1;
}
//-----------------------------------------------------------------------------
