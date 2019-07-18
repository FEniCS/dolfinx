// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <dolfin/mesh/cell_types.h>
#include <map>
#include <set>
#include <vector>

namespace dolfin
{

namespace fem
{

class ReferenceCellTopology
{
public:
  typedef int Edge[2];

  /// Get vertex indices of all edges
  static const Edge* get_edge_vertices(mesh::CellType cell_type);

  typedef int Face[4];

  /// Get vertex indices of all faces
  static const Face* get_face_vertices(mesh::CellType cell_type);

  /// Get edge indices of all faces
  static const Face* get_face_edges(mesh::CellType cell_type);

  typedef double Point[3];

  /// Get geometric points for all vertices
  static const Point* get_vertices(mesh::CellType cell_type);

  // Map from entity {dim_e, entity_e} to map {dim_c, (entities_c)}
  static std::map<std::array<int, 2>, std::vector<std::set<int>>>
  entity_closure(mesh::CellType cell_type);
};
} // namespace fem
} // namespace dolfin
