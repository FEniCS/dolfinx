// Copyright (C) 2006 Anders Logg
//
// This file is part of DOLFIN (https://www.fenicsproject.org)
//
// SPDX-License-Identifier:    LGPL-3.0-or-later

#include "Face.h"
#include "Cell.h"
#include <dolfin/geometry/Point.h>

using namespace dolfin;
using namespace dolfin::mesh;

//-----------------------------------------------------------------------------
double Face::area() const
{
  dolfin_assert(_mesh);
  dolfin_assert(_mesh->ordered());

  const std::size_t D = _mesh->topology().dim();

  // If the Face is the same topological dimension as cell
  if (D == 2)
  {
    // Get the cell corresponding to this Face
    const Cell cell(*_mesh, this->index());

    // Return the generalized volume (area)
    return cell.volume();
  }
  else
  {

    // Initialize needed connectivity
    _mesh->init(2, D);

    // Get cell to which face belong (first cell when there is more than one)
    const Cell cell(*_mesh, this->entities(D)[0]);

    // Get local index of facet with respect to the cell
    const std::size_t local_facet = cell.index(*this);

    return cell.facet_area(local_facet);
  }
}
//-----------------------------------------------------------------------------
geometry::Point Face::normal() const
{
  dolfin_assert(_mesh);
  dolfin_assert(_mesh->ordered());

  const std::size_t tD = _mesh->topology().dim();
  const std::size_t gD = _mesh->geometry().dim();

  // Check for when Cell has the same topological dimension as Face and we are
  // in R^2
  if (tD == 2 && gD == 2)
  {
    log::dolfin_error("Face.cpp", "compute Face normal",
                 "Don't know how to compute Face normal for a Face in a 2D "
                 "mesh embedded in R^2.");
  }

  // Check for when Cell has the same topological dimension as Face and we are
  // in R^3
  if (tD == 2 && gD == 3)
  {
    dolfin_not_implemented();
  }

  // Initialize needed connectivity
  _mesh->init(2, tD);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(tD)[0]);

  // Get local index of facet with respect to the cell
  const std::size_t local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------
