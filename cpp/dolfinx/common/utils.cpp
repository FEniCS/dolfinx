// Copyright (C) 2009 Anders Logg
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "utils.h"
#include <sstream>

//-----------------------------------------------------------------------------
std::string dolfinx::common::indent(std::string block)
{
  std::string indentation("  ");
  std::stringstream s;

  s << indentation;
  for (std::size_t i = 0; i < block.size(); ++i)
  {
    s << block[i];
    if (block[i] == '\n' && i < block.size() - 1)
      s << indentation;
  }

  return s.str();
}
//-----------------------------------------------------------------------------
std::shared_ptr<const dolfinx::common::IndexMap>
dolfinx::common::compress_index_map(
    std::shared_ptr<const dolfinx::common::IndexMap> map,
    const xtl::span<const std::int32_t>& indices)
{
  std::cout << map->size_local() << " " << indices.size() << "\n";
  return std::make_shared<const dolfinx::common::IndexMap>(
      map->comm(dolfinx::common::IndexMap::Direction::forward),
      map->size_local());
}
//-----------------------------------------------------------------------------
