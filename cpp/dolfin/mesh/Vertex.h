// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "MeshEntity.h"
#include <dolfin/common/types.h>

namespace dolfin
{
namespace mesh
{

/// A Vertex is a MeshEntity of topological dimension 0.

class Vertex : public MeshEntity
{
public:
  /// Create vertex on given mesh
  Vertex(const Mesh& mesh, std::int32_t index) : MeshEntity(mesh, 0, index) {}

  /// Create vertex from mesh entity
  Vertex(MeshEntity& entity) : MeshEntity(entity.mesh(), 0, entity.index()) {}

  /// Copy constructor
  Vertex(const Vertex& v) = default;

  /// Move constructor
  Vertex(Vertex&& v) = default;

  /// Destructor
  ~Vertex() = default;
};
} // namespace mesh
} // namespace dolfin
