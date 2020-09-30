// Copyright (C) 2019-2020 Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>
#include <dolfinx/graph/AdjacencyList.h>

namespace dolfinx
{
namespace fem
{
class ElementDofLayout;
}

namespace mesh
{
enum class CellType;
class Mesh;

/// Extract topology from cell data, i.e. extract cell vertices
/// @param[in] cell_type The cell shape
/// @param[in] layout The layout of geometry 'degrees-of-freedom' on the
///   reference cell
/// @param[in] cells List of 'nodes' for each cell using global indices.
///   The layout must be consistent with \p layout.
/// @return Cell topology. The global indices will, in general, have
///   'gaps' due to mid-side and other higher-order nodes being removed
///   from the input @p cell.
graph::AdjacencyList<std::int64_t>
extract_topology(const CellType& cell_type, const fem::ElementDofLayout& layout,
                 const graph::AdjacencyList<std::int64_t>& cells);

/// Compute (generalized) volume of mesh entities of given dimension
Eigen::ArrayXd volume_entities(const Mesh& mesh,
                               const Eigen::Ref<const Eigen::ArrayXi>& entities,
                               int dim);

/// Compute circumradius of mesh entities
Eigen::ArrayXd circumradius(const Mesh& mesh,
                            const Eigen::Ref<const Eigen::ArrayXi>& entities,
                            int dim);

/// Compute greatest distance between any two vertices
Eigen::ArrayXd h(const Mesh& mesh,
                 const Eigen::Ref<const Eigen::ArrayXi>& entities, int dim);

/// Compute inradius of cells
Eigen::ArrayXd inradius(const Mesh& mesh,
                        const Eigen::Ref<const Eigen::ArrayXi>& entities);

/// Compute dim*inradius/circumradius for given cells
Eigen::ArrayXd radius_ratio(const Mesh& mesh,
                            const Eigen::Ref<const Eigen::ArrayXi>& entities);

/// Compute normal to given cell (viewed as embedded in 3D)
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor>
cell_normals(const Mesh& mesh, int dim);

/// Compute midpoints or mesh entities of a given dimension
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> midpoints(
    const mesh::Mesh& mesh, int dim,
    const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& entities);

/// Compute indicies of all mesh entities that evaluate to true for the
/// provided geometric marking function. An entity is considered marked
/// if the marker function evaluates true for all of its vertices.
///
/// @param[in] mesh The mesh
/// @param[in] dim The topological dimension of the entities to be
///   considered
/// @param[in] marker The marking function
/// @returns List of marked entity indices, including any ghost indices
///   (indices local to the process)
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> locate_entities(
    const mesh::Mesh& mesh, const int dim,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker);

/// Compute indicies of all mesh entities that are attached to an owned
/// boundary facet and evaluate to true for the provided geometric
/// marking function. An entity is considered marked if the marker
/// function evaluates true for all of its vertices.
///
/// @note For vertices and edges, in parallel this function will not
/// necessarily mark all entities that are on the exterior boundary. For
/// example, it is possible for a process to have a vertex that lies on
/// the boundary without any of the attached facets being a boundary
/// facet. When used to find degrees-of-freedom, e.g. using
/// fem::locate_dofs_topological, the function that uses the data
/// returned by this function must typically perform some parallel
/// communication.
///
/// @param[in] mesh The mesh
/// @param[in] dim The topological dimension of the entities to be
///   considered. Must be less than the topological dimension of the
///   mesh.
/// @param[in] marker The marking function
/// @returns List of marked entity indices (indices local to the
///   process)
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> locate_entities_boundary(
    const mesh::Mesh& mesh, const int dim,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker);

/// Compute the geometry indices of vertices of the given entities from the mesh
/// geometry
/// @param[in] mesh Mesh
/// @param[in] dim Topological dimension of the entities of interest
/// @param[in] entity_list List of entity indices (local)
/// @param[in] orient If true, in 3D, reorients facets to have consistent normal
/// direction
/// @return Indices in the geometry array for the mesh entity vertices, i.e.
/// indices(i, j) is the position in the geometry array of the j-th vertex of
/// the entity entity_list[i].
Eigen::Array<std::int32_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
entities_to_geometry(
    const mesh::Mesh& mesh, const int dim,
    const Eigen::Array<std::int32_t, Eigen::Dynamic, 1>& entity_list,
    bool orient);

/// Compute the indices (local) of all exterior facets. An exterior facet
/// (co-dimension 1) is one that is connected globally to only one cell of
/// co-dimension 0).
/// @param[in] mesh Mesh
/// @return List of facet indices of exterior facets of the mesh
Eigen::Array<std::int32_t, Eigen::Dynamic, 1>
exterior_facet_indices(const Mesh& mesh);

} // namespace mesh
} // namespace dolfinx
