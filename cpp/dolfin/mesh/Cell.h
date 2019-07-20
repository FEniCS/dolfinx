// Copyright (C) 2006-2015 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CoordinateDofs.h"
#include "Geometry.h"
#include "Mesh.h"
#include "MeshEntity.h"
#include "MeshFunction.h"
#include "utils.h"
#include <Eigen/Dense>
#include <dolfin/common/types.h>
#include <memory>

namespace dolfin
{
namespace mesh
{

/// A Cell is a _MeshEntity_ of topological codimension 0.

class Cell : public MeshEntity
{
public:
  /// Create cell on given mesh with given index
  ///
  /// @param    mesh
  ///         The mesh.
  /// @param    index
  ///         The index.
  Cell(const Mesh& mesh, std::int32_t index)
      : MeshEntity(mesh, mesh.topology().dim(), index)
  {
  }

  /// Copy constructor
  Cell(const Cell& cell) = default;

  /// Move constructor
  Cell(Cell&& cell) = default;

  /// Destructor
  ~Cell() = default;

  /// Assignment operator
  Cell& operator=(const Cell& cell) = default;
};
} // namespace mesh
} // namespace dolfin
