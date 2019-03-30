// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

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
  static const int* num_entities(CellType cell_type);
  static int num_vertices(CellType cell_type);
  static int num_edges(CellType cell_type);
  static int num_facets(CellType cell_type);

  static CellType facet_type(CellType cell_type, int k=0);

  typedef int Edge[2];
  static const Edge* get_edges(CellType cell_type);

  typedef int Face[4];
  static const Face* get_faces(CellType cell_type);

  // Get connectivity from entities of dimension d0 to d1.
  static const int* get_entities(CellType cell_type, int d0, int d1);

  typedef double Point[3];
  static const Point* get_vertices(CellType cell_type);
};
} // namespace fem
} // namespace dolfin
