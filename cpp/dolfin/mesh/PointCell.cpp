// Copyright (C) 2007-2008 Kristian B. Oelgaard
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "PointCell.h"
#include "Cell.h"
#include "Facet.h"
#include "MeshEntity.h"
#include "Vertex.h"

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double PointCell::squared_distance(const Cell& cell,
                                   const Eigen::Vector3d& point) const
{
  throw std::runtime_error("Not defined");
  return 0.0;
}
//-----------------------------------------------------------------------------
