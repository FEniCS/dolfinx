// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once
#include <cstdint>
#include <dolfinx/mesh/Mesh.h>
#include <utility>
#include <vector>

namespace dolfinx
{
namespace mesh
{
class Mesh;
} // namespace mesh

namespace io
{
std::pair<Eigen::Array<std::int32_t, Eigen::Dynamic, 1>,
          Eigen::Array<std::int8_t, Eigen::Dynamic, 1>>
create_pyvista_topology(const mesh::Mesh& mesh, int dim,
                        std::vector<std::int32_t>& entities);
} // namespace io
} // namespace dolfinx