// Copyright (C) 2019-2026 Garth N. Wells
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <array>
#include <basix/cell.h>
#include <dolfinx/graph/AdjacencyList.h>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace dolfinx::mesh
{

/// Cell type identifier
enum class CellType : std::int8_t
{
  // NOTE: Simplex cells have index > 0, see mesh::is_simplex
  point = 1,
  interval = 2,
  triangle = 3,
  tetrahedron = 4,
  quadrilateral = -4,
  pyramid = -5,
  prism = -6,
  hexahedron = -8
};

/// Get the cell string type for a cell type
/// @param[in] type The cell type
/// @return The cell type string
std::string to_string(CellType type);

/// Get the cell type from a cell string
/// @param[in] cell Cell shape string
/// @return The cell type
CellType to_type(const std::string& cell);

/// Return topological dimension of cell type
inline int cell_dim(CellType type)
{
  switch (type)
  {
  case CellType::point:
    return 0;
  case CellType::interval:
    return 1;
  case CellType::triangle:
    return 2;
  case CellType::quadrilateral:
    return 2;
  case CellType::tetrahedron:
    return 3;
  case CellType::hexahedron:
    return 3;
  case CellType::prism:
    return 3;
  case CellType::pyramid:
    return 3;
  default:
    throw std::runtime_error("Unsupported cell type");
  }
}

/// @brief Return facet type of cell.
///
/// For simplex and hypercube cell types, this is independent of the
/// facet index, but for prism and pyramid, it can be triangle or
/// quadrilateral.
///
/// @param[in] type Cell type.
/// @param[in] index Facet index (relative to the cell).
/// @return Type of facet for this cell at this index.
inline CellType cell_facet_type(CellType type, int index)
{
  switch (type)
  {
  case CellType::point:
    return CellType::point;
  case CellType::interval:
    return CellType::point;
  case CellType::triangle:
    return CellType::interval;
  case CellType::tetrahedron:
    return CellType::triangle;
  case CellType::quadrilateral:
    return CellType::interval;
  case CellType::pyramid:
    if (index == 0)
      return CellType::quadrilateral;
    else
      return CellType::triangle;
  case CellType::prism:
    if (index == 0 or index == 4)
      return CellType::triangle;
    else
      return CellType::quadrilateral;
  case CellType::hexahedron:
    return CellType::quadrilateral;
  default:
    throw std::runtime_error("Unknown cell type.");
  }
}

/// Return type of cell for entity of dimension d at given entity index.
inline CellType cell_entity_type(CellType type, int d, int index)
{
  if (int dim = mesh::cell_dim(type); d == dim)
    return type;
  else if (d == 1)
    return CellType::interval;
  else if (d == (dim - 1))
    return mesh::cell_facet_type(type, index);
  else
    return CellType::point;
}

/// Return list of entities, where entities(e, k) is the local vertex
/// index for the kth vertex of entity e of dimension dim
graph::AdjacencyList<std::vector<int>> get_entity_vertices(CellType type, int dim);

/// Get entities of dimension dim1 and that make up entities of dimension
/// dim0.
graph::AdjacencyList<std::vector<int>> get_sub_entities(CellType type, int dim0, int dim1);

/// @brief Number of entities of dimension.
///
/// @param[in] dim Entity dimension.
/// @param[in] type Cell type.
/// @return Number of entities in cell.
int cell_num_entities(CellType type, int dim);

/// @brief Check if cell is a simplex.
///
/// @param[in] type Cell type.
/// @return True is the cell type is a simplex.
bool is_simplex(CellType type);

/// @brief Number vertices for a cell type.
///
/// @param[in] type Cell type
/// @return Number of cell vertices
int num_cell_vertices(CellType type);

// [dim, entity] -> closure{sub_dim, (sub_entities)}

/// Closure entities for a cell, i.e., all lower-dimensional entities
/// attached to a cell entity. Map from entity {dim_e, entity_e} to
/// closure{sub_dim, (sub_entities)}
std::map<std::array<int, 2>, std::vector<std::set<int>>>
cell_entity_closure(CellType cell_type);

/// Convert a cell type to a Basix cell type
basix::cell::type cell_type_to_basix_type(CellType celltype);

/// Get a cell type from a Basix cell type
CellType cell_type_from_basix_type(basix::cell::type celltype);

} // namespace dolfinx::mesh
