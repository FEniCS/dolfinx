// Copyright (C) 2006-2010 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshGeometry.h"
#include <dolfin/geometry/Point.h>

namespace dolfin
{

/// A Vertex is a MeshEntity of topological dimension 0.

class Vertex : public MeshEntity
{
public:
  /// Create vertex on given mesh
  Vertex(const Mesh& mesh, std::size_t index) : MeshEntity(mesh, 0, index) {}

  /// Create vertex from mesh entity
  Vertex(MeshEntity& entity) : MeshEntity(entity.mesh(), 0, entity.index()) {}

  /// Destructor
  ~Vertex() {}

  /// Return value of vertex coordinate i
  double x(std::size_t i) const { return _mesh->geometry().x(_local_index, i); }

  /// Return vertex coordinates as a 3D point value
  Point point() const { return _mesh->geometry().point(_local_index); }

  /// Return array of vertex coordinates (const version)
  const double* x() const { return _mesh->geometry().x(_local_index); }
};

}
