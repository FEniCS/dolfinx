// Copyright (C) 2008-2018 Anders Logg, Ola Skavhaug and Garth N. Wells
//
// This file is part of DOLFINX (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <dolfinx/graph/AdjacencyList.h>
#include <memory>
#include <mpi.h>
#include <tuple>
#include <vector>

namespace dolfinx
{

namespace common
{
class IndexMap;
}

namespace mesh
{
class Topology;
} // namespace mesh

namespace fem
{
class ElementDofLayout;

/// Build dofmap data for an element on a mesh topology
/// @param[in] comm MPI communicator
/// @param[in] topology The mesh topology
/// @param[in] element_dof_layout The element dof layout for the function
/// space
/// @return The index map and local to global DOF data for the DOF map.
std::tuple<std::shared_ptr<common::IndexMap>, int,
           graph::AdjacencyList<std::int32_t>>
build_dofmap_data(MPI_Comm comm, const mesh::Topology& topology,
                  const ElementDofLayout& element_dof_layout);

} // namespace fem
} // namespace dolfinx
