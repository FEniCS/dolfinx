// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "Mesh.h"
#include "MeshEntity.h"
#include "Topology.h"
#include <utility>

namespace dolfin
{

namespace mesh
{

/// A Facet is a MeshEntity of topological codimension 1.

class Facet : public MeshEntity
{
public:
  /// Constructor
  Facet(const Mesh& mesh, std::int32_t index)
      : MeshEntity(mesh, mesh.topology().dim() - 1, index)
  {
    // Do nothing
  }

  /// Destructor
  ~Facet() = default;
};
} // namespace mesh
} // namespace dolfin
