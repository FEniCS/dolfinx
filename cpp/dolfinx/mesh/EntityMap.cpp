// Copyright (C) 2025 Jørgen S. Dokken and Joseph P. Dean
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "EntityMap.h"
#include "Topology.h"
#include <span>
#include <unordered_map>
#include <vector>

namespace dolfinx::mesh
{
//-----------------------------------------------------------------------------
std::size_t EntityMap::dim() const { return _dim; }
//-----------------------------------------------------------------------------
std::shared_ptr<const Topology> EntityMap::topology() const
{
  return _topology;
}
//-----------------------------------------------------------------------------
std::shared_ptr<const Topology> EntityMap::sub_topology() const
{
  return _sub_topology;
}

//-----------------------------------------------------------------------------
} // namespace dolfinx::mesh
