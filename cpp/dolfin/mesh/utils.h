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
enum class CellType : int
{
  // NOTE: Simplex cell have index > 0, see mesh::is_simplex.
  point = 1,
  interval = 2,
  triangle = 3,
  tetrahedron = 4,
  quadrilateral = -4,
  hexahedron = -8
};

/// Convert from cell type to string
std::string to_string(CellType type);

/// Convert from string to cell type
CellType to_type(std::string type);

/// Check if cell is a simplex
bool is_simplex(CellType type);

/// Check if cell is a simplex
int num_cell_vertices(CellType type);

} // namespace mesh
} // namespace dolfin
