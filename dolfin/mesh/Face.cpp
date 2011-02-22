// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2011.
//
// First added:  2006-06-02
// Last changed: 2011-02-22

#include "Cell.h"
#include "Point.h"
#include "Face.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Face::area() const
{
  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(3)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.facet_area(local_facet);
}
//-----------------------------------------------------------------------------
double Face::normal(uint i) const
{
  _mesh->init(2);
  _mesh->init(2, 3);
  assert(_mesh->ordered());

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(3)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet, i);
}
//-----------------------------------------------------------------------------
Point Face::normal() const
{
  _mesh->init(2);
  _mesh->init(2, 3);
  assert(_mesh->ordered());

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(3)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------

