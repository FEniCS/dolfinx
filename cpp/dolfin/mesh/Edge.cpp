// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Edge.h"
#include <cmath>
#include <dolfin/mesh/Geometry.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double Edge::length() const
{
  const std::int32_t* vertices = entities(0);
  assert(vertices);
  const Eigen::Vector3d p0 = mesh().geometry().x(vertices[0]);
  const Eigen::Vector3d p1 = mesh().geometry().x(vertices[1]);
  return (p0 - p1).norm();
}
//-----------------------------------------------------------------------------
