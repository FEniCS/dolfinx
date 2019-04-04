// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <map>
#include <set>
#include <vector>

namespace dolfin
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

namespace fem
{

class ReferenceCellTopology
{
public:
  /// Topological dimension of cells of type cell_type
  static int dim(CellType cell_type);

  /// Number of entities of each topological dimension (0, 1, 2, 3)
  /// for cells of type cell_type
  static const int* num_entities(CellType cell_type);

  /// Get CellType of subentity of this cell_type, which has dimension dim.
  /// FIXME: currently k does nothing.
  static CellType entity_type(CellType cell_type, int dim, int k = 0);

  /// Get CellType of facets of cell_type, e.g. for tetrahedron, this is
  /// triangle. Equivalent to entity_type(cell_type, dim(cell_type) -1)
  /// FIXME: currently k does nothing.
  static CellType facet_type(CellType cell_type, int k = 0);

  typedef int Edge[2];

  /// Get vertex indices of all edges
  static const Edge* get_edge_vertices(CellType cell_type);

  typedef int Face[4];

  /// Get vertex indices of all faces
  static const Face* get_face_vertices(CellType cell_type);

  /// Get edge indices of all faces
  static const Face* get_face_edges(CellType cell_type);

  typedef double Point[3];

  /// Get geometric points for all vertices
  static const Point* get_vertices(CellType cell_type);

  // Map from entity {dim_e, entity_e} to map {dim_c, (entities_c)}
  static std::map<std::array<int, 2>, std::vector<std::set<int>>>
  entity_closure(CellType cell_type);
};
} // namespace fem
} // namespace dolfin
