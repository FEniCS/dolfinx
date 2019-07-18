// Copyright (C) 2006-2017 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include "CellType.h"
#include <Eigen/Dense>
#include <string>
#include <vector>

namespace dolfin
{

namespace mesh
{
class Cell;
class MeshEntity;
template <typename T>
class MeshFunction;

/// This class implements functionality for interval cell meshes.

class IntervalCell : public mesh::CellTypeOld
{
public:
  /// Specify cell type and facet type
  IntervalCell() : mesh::CellTypeOld(CellType::interval) {}
};
} // namespace mesh
} // namespace dolfin
