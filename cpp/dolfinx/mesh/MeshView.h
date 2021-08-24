// Copyright (C) 2021 Jørgen S. Dokken
//
// This file is part of DOLFINx (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once
#include "Mesh.h"
#include "Topology.h"
#include <dolfinx/graph/AdjacencyList.h>
#include <xtl/xspan.hpp>

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
  MeshView(const std::shared_ptr<const Mesh> parent_mesh, int dim,
           tcb::span<std::int32_t> entities);

  /// Create a mesh-view for all entities of a given topological dimension
  /// @param[in] parent_mesh The parent mesh
  /// @param[in] dim The dimension of the entities to make a view of
  // MeshView(const std::shared_ptr<const Mesh> parent_mesh, int dim);

  /// Return the topology of the mesh
  std::shared_ptr<Topology> topology() { return _topology; };

  /// Return the dimension of the mesh view
  std::int32_t dim() { return _dim; };

  /// Return a pointer to the parent mesh
  std::shared_ptr<const Mesh> parent_mesh() { return _parent_mesh; };

  /// Return map from child entities to parent mesh entities (local to process)
  const std::vector<std::int32_t>& parent_entities()
  {
    return _parent_entity_map;
  }

  /// Return map from child vertices to parent mesh vertices (local to process)
  const std::vector<std::int32_t>& parent_vertices()
  {
    return _parent_vertex_map;
  }

  /// Return the geometry dofmap for the MeshView
  const std::shared_ptr<graph::AdjacencyList<std::int32_t>> geometry_dofmap()
  {
    return _geom_dofmap;
  }

private:
  std::shared_ptr<const Mesh> _parent_mesh;
  std::vector<std::int32_t> _parent_entity_map;
  std::vector<std::int32_t> _parent_vertex_map;
  const std::int32_t _dim;
  std::shared_ptr<Topology> _topology;
  std::shared_ptr<graph::AdjacencyList<std::int32_t>> _geom_dofmap;
};

} // namespace dolfinx::mesh