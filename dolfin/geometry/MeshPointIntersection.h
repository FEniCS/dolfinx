// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>
#include <vector>

namespace dolfin
{

// Forward declarations
class Mesh;
class Point;

/// This class represents an intersection between a _Mesh_ and a
/// _Point_. The resulting intersection is stored as a list of zero
/// or more cells.

class MeshPointIntersection
{
public:
  /// Compute intersection between mesh and point
  MeshPointIntersection(const Mesh& mesh, const Point& point);

  /// Destructor
  ~MeshPointIntersection();

  /// Return the list of (local) indices for intersected cells
  const std::vector<unsigned int>& intersected_cells() const
  {
    return _intersected_cells;
  }

private:
  // The list of (local) indices for intersected cells
  std::vector<unsigned int> _intersected_cells;
};
}


