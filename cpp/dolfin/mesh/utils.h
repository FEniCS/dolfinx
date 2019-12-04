// Copyright (C) 2019 Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <Eigen/Dense>

namespace dolfin
{
namespace mesh
{
class Mesh;
class MeshEntity;

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

/// TODO
Eigen::Array<std::int32_t, Eigen::Dynamic, 1> compute_marked_entities(
    const mesh::Mesh& mesh,
    const int dim,
    const std::function<Eigen::Array<bool, Eigen::Dynamic, 1>(
        const Eigen::Ref<const Eigen::Array<double, 3, Eigen::Dynamic,
                                            Eigen::RowMajor>>&)>& marker,
    bool boundary_only = true);

} // namespace mesh
} // namespace dolfin
