// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <array>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace dolfin
{
namespace mesh
{
class Cell;
class Mesh;
class MeshEntity;

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

/// Return type of cell for entity of dimension d
CellType cell_entity_type(CellType type, int d);

/// Return facet type of cell
CellType cell_facet_type(CellType type);

/// Return array entities(num entities, num vertices per entity), where
/// entities(e, k) is the local vertex index for the kth vertex of
/// entity e of dimension dim
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_entity_vertices(CellType type, int dim);

// Get entities of dimsion dim1 and that make up entities of dimension
// dim0
Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
get_sub_entities(CellType type, int dim0, int dim1);

/// Return topological dimension of cell type
int cell_dim(CellType type);

int cell_num_entities(mesh::CellType type, int dim);

/// Check if cell is a simplex
bool is_simplex(CellType type);

/// Num vertices for a cell type
int num_cell_vertices(CellType type);

  // [dim, entity] -> closure{sub_dim, (sub_entities)}

/// Closure entities for a cell, i.e., all lower-dimensional entities
/// attached to a cell entity. Map from entity {dim_e, entity_e} to
/// closure{sub_dim, (sub_entities)}
std::map<std::array<int, 2>, std::vector<std::set<int>>>
cell_entity_closure(mesh::CellType cell_type);

/// Mapping of DOLFIN/UFC vertex ordering to VTK/XDMF ordering
std::vector<std::int8_t> vtk_mapping(CellType type);

} // namespace mesh
} // namespace dolfin
