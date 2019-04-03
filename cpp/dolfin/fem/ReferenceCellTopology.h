// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <map>
#include <set>

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
  static int dim(CellType cell_type);
  static const int* num_entities(CellType cell_type);

  // Get entity type of dimension d
  static CellType entity_type(CellType cell_type, int dim, int k = 0);

  static CellType facet_type(CellType cell_type, int k = 0);

  typedef int Edge[2];
  static const Edge* get_edge_vertices(CellType cell_type);

  typedef int Face[4];
  static const Face* get_face_vertices(CellType cell_type);
  static const Face* get_face_edges(CellType cell_type);

  // Get connectivity from entities of dimension d0 to d1.
  // static const int* get_entities(CellType cell_type, int d0, int d1);

  typedef double Point[3];
  static const Point* get_vertices(CellType cell_type);

  // Map from entity {dim_e, entity_e} to map {dim_c, (entities_c)}
  static std::map<std::array<int, 2>, std::map<int, std::set<int>>>
  entity_closure(CellType cell_type);
};
} // namespace fem
} // namespace dolfin
