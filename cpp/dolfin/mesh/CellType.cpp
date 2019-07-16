// Copyright (C) 2006-2011 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "CellType.h"
#include "Cell.h"
#include "HexahedronCell.h"
#include "IntervalCell.h"
#include "PointCell.h"
#include "QuadrilateralCell.h"
#include "TetrahedronCell.h"
#include "TriangleCell.h"
#include "Vertex.h"
#include <algorithm>
#include <array>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
CellTypeOld::CellTypeOld(CellType cell_type) : type(cell_type) {}
//-----------------------------------------------------------------------------
CellTypeOld* CellTypeOld::create(CellType type)
{
  switch (type)
  {
  case CellType::point:
    return new PointCell();
  case CellType::interval:
    return new IntervalCell();
  case CellType::triangle:
    return new TriangleCell();
  case CellType::tetrahedron:
    return new TetrahedronCell();
  case CellType::quadrilateral:
    return new QuadrilateralCell();
  case CellType::hexahedron:
    return new HexahedronCell();
  default:
    throw std::runtime_error("Unknown cell type");
  }

  return nullptr;
}
//-----------------------------------------------------------------------------
