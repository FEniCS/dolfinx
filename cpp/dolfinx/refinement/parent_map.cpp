// Copyright (C) 2021 Matthew Scroggs
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "parent_map.h"

using namespace dolfinx;
using namespace refinement;

//-----------------------------------------------------------------------------
ParentRelationshipInfo::ParentRelationshipInfo(
    std::vector<std::pair<std::int8_t, std::int64_t>> parent_map)
    : _parent_map(parent_map)
{
}
//-----------------------------------------------------------------------------
std::vector<std::pair<std::int8_t, std::int64_t>>
ParentRelationshipInfo::parent_map() const
{
  return _parent_map;
}
//-----------------------------------------------------------------------------
