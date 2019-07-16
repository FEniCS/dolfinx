// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
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

/// Return topological dimension of cell type
int cell_dim(CellType type);

int cell_num_entities(mesh::CellType type, int dim);

/// Check if cell is a simplex
bool is_simplex(CellType type);

/// Num vertices for a cell type
int num_cell_vertices(CellType type);

/// Compute (generalized) volume of mesh entities of given dimension
Eigen::ArrayXd volume_cells(const Mesh& mesh,
                            const Eigen::Ref<const Eigen::ArrayXi> entities);

/// Compute (generalized) volume of mesh entities of given dimension
Eigen::ArrayXd volume_entities(const Mesh& mesh,
                               const Eigen::Ref<const Eigen::ArrayXi> entities,
                               int dim);

/// Compute (generalized) volume of mesh entity. Note: this function is
/// not very efficient. Use the vectorised version for computing
/// multiple volumes.
double volume(const MeshEntity& e);

/// Compute circumradius of mesh entities
Eigen::ArrayXd circumradius(const Mesh& mesh,
                            const Eigen::Ref<const Eigen::ArrayXi> entities,
                            int dim);

/// Compute greatest distance between any two vertices
Eigen::ArrayXd h(const Mesh& mesh,
                 const Eigen::Ref<const Eigen::ArrayXi> entities, int dim);

/// Compute inradius of cells
Eigen::ArrayXd inradius(const Mesh& mesh,
                        const Eigen::Ref<const Eigen::ArrayXi> entities);

/// Compute dim*inradius/circumradius for given cells
Eigen::ArrayXd radius_ratio(const Mesh& mesh,
                            const Eigen::Ref<const Eigen::ArrayXi> entities);

/// Mapping of DOLFIN/UFC vertex ordering to VTK/XDMF ordering
std::vector<std::int8_t> vtk_mapping(CellType type);

} // namespace mesh
} // namespace dolfin
