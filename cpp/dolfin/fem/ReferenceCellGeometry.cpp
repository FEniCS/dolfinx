// Copyright (C) 2019 Chris Richardson and Garth N. Wells
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "ReferenceCellGeometry.h"
#include <cassert>
#include <stdexcept>

using namespace dolfin;
using namespace dolfin::fem;

//-----------------------------------------------------------------------------
Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
ReferenceCellGeometry::get_vertices(mesh::CellType cell_type)
{
  const static Eigen::Array<double, 1, 1> point
      = (Eigen::Array<double, 1, 1>() << 0.0).finished();
  const static Eigen::Array<double, 2, 1> interval
      = (Eigen::Array<double, 2, 1>() << 0.0, 1.0).finished();
  const static Eigen::Array<double, 3, 2, Eigen::RowMajor> triangle
      = (Eigen::Array<double, 3, 2, Eigen::RowMajor>() << 0, 0, 1, 0, 0, 1)
            .finished();
  const static Eigen::Array<double, 4, 2, Eigen::RowMajor> quadrilateral
      = (Eigen::Array<double, 4, 2, Eigen::RowMajor>() << 0, 0, 0, 1, 1, 0, 1,
         1)
            .finished();
  const static Eigen::Array<double, 4, 3, Eigen::RowMajor> tetrahedron
      = (Eigen::Array<double, 4, 3, Eigen::RowMajor>() << 0, 0, 0, 1, 0, 0, 0,
         1, 0, 0, 0, 1)
            .finished();
  const static Eigen::Array<double, 6, 4, Eigen::RowMajor> hexahedron
      = (Eigen::Array<double, 6, 4, Eigen::RowMajor>() << 0, 0, 0, 0, 0, 1, 0,
         1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1)
            .finished();

  switch (cell_type)
  {
  case mesh::CellType::point:
    return point;
  case mesh::CellType::interval:
    return interval;
  case mesh::CellType::triangle:
    return triangle;
  case mesh::CellType::quadrilateral:
    return quadrilateral;
  case mesh::CellType::tetrahedron:
    return tetrahedron;
  case mesh::CellType::hexahedron:
    return hexahedron;
  default:
    throw std::runtime_error("Unknown cell type.");
  }

  return Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic,
                      Eigen::RowMajor>();
}
//-----------------------------------------------------------------------------
