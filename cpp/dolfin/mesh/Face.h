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

/// A Face is a MeshEntity of topological dimension 2.

class Face : public MeshEntity
{
public:
  /// Constructor
  Face(const Mesh& mesh, std::int32_t index) : MeshEntity(mesh, 2, index) {}

  /// Destructor
  ~Face() = default;

  /// Compute normal to the face
  Eigen::Vector3d normal() const;
};
} // namespace mesh
} // namespace dolfin
