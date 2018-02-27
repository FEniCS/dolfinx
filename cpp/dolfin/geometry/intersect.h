// Copyright (C) 2013 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#pragma once

#include <memory>

namespace dolfin
{

// Forward declarations
class Mesh;
class Point;
class MeshPointIntersection;

namespace geometry
{

/// Compute and return intersection between _Mesh_ and _Point_.
///
/// *Arguments*
///     mesh (_Mesh_)
///         The mesh to be intersected.
///     point (_Point_)
///         The point to be intersected.
///
/// *Returns*
///     _MeshPointIntersection_
///         The intersection data.
std::shared_ptr<const MeshPointIntersection> intersect(const Mesh& mesh,
                                                       const Point& point);
}
}
