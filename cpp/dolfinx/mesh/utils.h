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
class Mesh;
class MeshEntity;

/// Extract topology from cell data
/// @param[in] layout The layout of geometry 'degrees-of-freedom' on the
///     reference cell
/// @param[in] cells List of 'nodes' for each cell using global indices.
///     The layout must be consistent with \p layout.
/// @return Cell topology, with the global index for the vertices, order
///     v0, v1, v1, etc. The global indices will, in general, have
///     'gaps' due to mid-side and other higher-order nodes being
///     removed.
graph::AdjacencyList<std::int64_t>
extract_topology(const fem::ElementDofLayout& layout,
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

/// Compute of given facet with respect to the cell
Eigen::Vector3d normal(const MeshEntity& cell, int facet_local);

/// Compute midpoints or mesh entities of a given dimension
Eigen::Array<double, Eigen::Dynamic, 3, Eigen::RowMajor> midpoints(
    const mesh::Mesh& mesh, int dim,
    const Eigen::Ref<const Eigen::Array<int, Eigen::Dynamic, 1>>& entities);

/// Compute indicies (local to the process) of all mesh entities that
/// evaluate to true for the provided marking function. An entity is
/// considered marked if the marker function evaluates true for all of
/// the entities vertices.
/// @cond Work around doxygen bug for std::function
/// @param[in] mesh The mesh
/// @param[in] dim The topological dimension of the entities to be
///                considered
/// @param[in] marker The marking function
/// @returns List of marked entity indices (indices local to the
/// process)
/// @endcond
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> compute_marked_boundary_entities(
    const mesh::Mesh& mesh, const int dim,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker);

} // namespace mesh
} // namespace dolfinx
