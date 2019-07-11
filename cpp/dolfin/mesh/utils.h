// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <string>

namespace dolfin
{
namespace mesh
{
enum class CellType : short int
{
  point,
  interval,
  triangle,
  quadrilateral,
  tetrahedron,
  hexahedron
};

/// Convert from cell type to string
std::string to_string(CellType type);

/// Convert from string to cell type
CellType to_type(std::string type);

} // namespace mesh
} // namespace dolfin
