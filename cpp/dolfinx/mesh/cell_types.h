// Copyright (C) 2019-2020 Garth N. Wells
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
enum class CellType : int
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

/// Return type of cell for entity of dimension d at given entity index.
CellType cell_entity_type(CellType type, int d, int index);

/// Return facet type of cell
/// For simplex and hypercube cell types, this is independent of the facet
/// index, but for prism and pyramid, it can be triangle or quadrilateral.
/// @param[in] type The cell type
/// @param[in] index The facet index
/// @return The type of facet for this cell at this index
CellType cell_facet_type(CellType type, int index);

/// Return list of entities, where entities(e, k) is the local vertex
/// index for the kth vertex of entity e of dimension dim
graph::AdjacencyList<int> get_entity_vertices(CellType type, int dim);

/// Get entities of dimension dim1 and that make up entities of dimension
/// dim0
graph::AdjacencyList<int> get_sub_entities(CellType type, int dim0, int dim1);

/// Return topological dimension of cell type
int cell_dim(CellType type);

/// Number of entities of dimension dim
/// @param[in] dim Entity dimension
/// @param[in] type Cell type
/// @return Number of entities in cell
int cell_num_entities(CellType type, int dim);

/// Check if cell is a simplex
/// @param[in] type Cell type
/// @return True is the cell type is a simplex
bool is_simplex(CellType type);

/// Number vertices for a cell type
/// @param[in] type Cell type
/// @return The number of cell vertices
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
