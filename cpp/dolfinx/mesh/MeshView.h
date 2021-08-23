// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once
#include "Mesh.h"
#include "xtl/xspan.hpp"

namespace dolfinx::mesh
{

/// A MeshView consist of a set of of entities of any co-dimension of a parent
/// mesh. A mesh-view has its own topology, and two maps, one to map the local
/// cell index to the parent mesh, and one to map the local vertex index to a
/// parent vertex
class MeshView
{

public:
  /// Create a mesh-view
  /// @param[in] parent_mesh The parent mesh
  /// @param[in] dim The dimension of the entities to make a view of
  /// @param[in] entities List of local entities in the view
  MeshView(std::shared_ptr<const Mesh> parent_mesh, int dim,
           tcb::span<std::int32_t> entities);

private:
  std::shared_ptr<const Mesh> _parent_mesh;
};

} // namespace dolfinx::mesh