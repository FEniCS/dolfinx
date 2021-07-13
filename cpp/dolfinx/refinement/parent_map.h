// Copyright (C) 2021 Matthew Scroggs
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace dolfinx
{

namespace refinement
{

/// A class containing information about the relationship between a refined mesh
/// and its parent.
class ParentRelationshipInfo
{
public:
  /// Create a parent relationship
  ParentRelationshipInfo(
      std::vector<std::pair<std::int8_t, std::int64_t>> parent_map);

  /// Parent map
  std::vector<std::pair<std::int8_t, std::int64_t>> parent_map() const;

private:
  std::vector<std::pair<std::int8_t, std::int64_t>> _parent_map;
};

} // namespace refinement
} // namespace dolfinx
