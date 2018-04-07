// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshGeometry.h"
#include <dolfin/common/types.h>
#include <dolfin/geometry/Point.h>

namespace dolfin
{
namespace mesh
{

/// A Vertex is a MeshEntity of topological dimension 0.

class Vertex : public MeshEntity
{
public:
  /// Create vertex on given mesh
  Vertex(const Mesh& mesh, std::size_t index) : MeshEntity(mesh, 0, index) {}

  /// Create vertex from mesh entity
  Vertex(MeshEntity& entity) : MeshEntity(entity.mesh(), 0, entity.index()) {}

  /// Copy constructor
  Vertex(const Vertex& v) = default;

  /// Move constructor
  Vertex(Vertex&& v) = default;

  /// Destructor
  ~Vertex() = default;

  /// Return vertex coordinates as a 3D point value
  geometry::Point point() const
  {
    return _mesh->geometry().point(_local_index);
  }

  /// Return array of vertex coordinates (const version)
  Eigen::Ref<const EigenRowArrayXd> x() const
  {
    return _mesh->geometry().points().row(_local_index);
  }
};
}
}