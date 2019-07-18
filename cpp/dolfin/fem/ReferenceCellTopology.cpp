// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ReferenceCellTopology.h"
#include <cassert>
#include <stdexcept>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
const ReferenceCellTopology::Point*
ReferenceCellTopology::get_vertices(mesh::CellType cell_type)
{
  static const double interval[][3] = {{0.0}, {1.0}};
  static const double triangle[][3] = {{0.0, 0.0}, {1.0, 0.0}, {0.0, 1.0}};
  static const double quadrilateral[][3]
      = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
  static const double tetrahedron[][3]
      = {{0.0, 0.0, 0.0}, {1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
  static const double hexahedron[][3]
      = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.0}, {0.0, 1.0, 0.0}, {0.0, 1.0, 1.0},
         {1.0, 0.0, 0.0}, {1.0, 0.0, 1.0}, {1.0, 1.0, 0.0}, {1.0, 1.0, 1.0}};

  switch (cell_type)
  {
  case mesh::CellType::point:
    return nullptr;
  case mesh::CellType::interval:
    return interval;
  case mesh::CellType::triangle:
    return triangle;
  case mesh::CellType::quadrilateral:
    return quadrilateral;
  case mesh::CellType::tetrahedron:
    return tetrahedron;
  case mesh::CellType::hexahedron:
    return hexahedron;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return nullptr;
}
//-----------------------------------------------------------------------------
std::map<std::array<int, 2>, std::vector<std::set<int>>>
ReferenceCellTopology::entity_closure(mesh::CellType cell_type)
{
  const int cell_dim = mesh::cell_dim(cell_type);
  std::array<int, 4> num_entities{};
  for (int i = 0; i <= cell_dim; ++i)
    num_entities[i] = mesh::cell_num_entities(cell_type, i);

  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      edge_v = mesh::create_entities(cell_type, 1);
  const Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
      face_e = mesh::get_sub_entities(cell_type, 2, 1);

  std::map<std::array<int, 2>, std::vector<std::set<int>>> entity_closure;
  for (int dim = 0; dim <= cell_dim; ++dim)
  {
    for (int entity = 0; entity < num_entities[dim]; ++entity)
    {
      // Add self
      entity_closure[{{dim, entity}}].resize(cell_dim + 1);
      entity_closure[{{dim, entity}}][dim].insert(entity);

      if (dim == 3)
      {
        // Add all sub-entities
        for (int f = 0; f < num_entities[2]; ++f)
          entity_closure[{{dim, entity}}][2].insert(f);
        for (int e = 0; e < num_entities[1]; ++e)
          entity_closure[{{dim, entity}}][1].insert(e);
        for (int v = 0; v < num_entities[0]; ++v)
          entity_closure[{{dim, entity}}][0].insert(v);
      }

      if (dim == 2)
      {
        mesh::CellType face_type = mesh::cell_entity_type(cell_type, 2);
        const int num_edges = mesh::cell_num_entities(face_type, 1);
        for (int e = 0; e < num_edges; ++e)
        {
          // Add edge
          const int edge_index = face_e(entity, e);
          entity_closure[{{dim, entity}}][1].insert(edge_index);
          for (int v = 0; v < 2; ++v)
          {
            // Add vertex connected to edge
            entity_closure[{{dim, entity}}][0].insert(edge_v(edge_index, v));
          }
        }
      }

      if (dim == 1)
      {
        entity_closure[{{dim, entity}}][0].insert(edge_v(entity, 0));
        entity_closure[{{dim, entity}}][0].insert(edge_v(entity, 1));
      }
    }
  }

  return entity_closure;
}
//-----------------------------------------------------------------------------
