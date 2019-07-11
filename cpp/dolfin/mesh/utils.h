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
enum class CellTypeNew : int
{
  point,
  interval,
  triangle,
  quadrilateral,
  tetrahedron,
  hexahedron
};

  /// Convert from cell type to string
  std::string to_string(CellTypeNew type);


  /// Convert from string to cell type
  CellTypeNew to_type(std::string type);

} // namespace mesh
} // namespace dolfin
