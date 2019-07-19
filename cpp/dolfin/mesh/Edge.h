// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "MeshEntity.h"

namespace dolfin
{

namespace mesh
{
class Mesh;

/// An Edge is a _MeshEntity_ of topological dimension 1.

class Edge : public MeshEntity
{
public:
  /// Create edge on given mesh
  ///
  /// @param    mesh (_Mesh_)
  ///         The mesh.
  /// @param    index (std::size_t)
  ///         Index of the edge.
  Edge(const Mesh& mesh, std::int32_t index) : MeshEntity(mesh, 1, index) {}

  /// Create edge from mesh entity
  ///
  /// @param    entity (_MeshEntity_)
  ///         The mesh entity to create an edge from.
  Edge(MeshEntity& entity) : MeshEntity(entity.mesh(), 1, entity.index()) {}

  /// Destructor
  ~Edge() = default;

  /// Compute Euclidean length of edge
  ///
  /// @return     double
  ///         Euclidean length of edge.
  ///
  /// @code{.cpp}
  ///
  ///         UnitSquare mesh(2, 2);
  ///         Edge edge(mesh, 0);
  ///         log::info("%g", edge.length());
  ///
  /// @endcode
  double length() const;

};
} // namespace mesh
} // namespace dolfin
