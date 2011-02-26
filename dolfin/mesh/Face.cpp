// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Garth N. Wells, 2011.
//
// First added:  2006-06-02
// Last changed: 2011-02-26

#include "Cell.h"
#include "Point.h"
#include "Face.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
double Face::area() const
{
  assert(_mesh);
  assert(_mesh->ordered());

  // Initialize needed connectivity
  const uint D = _mesh->topology().dim();
  _mesh->init(2, D);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.facet_area(local_facet);
}
//-----------------------------------------------------------------------------
double Face::normal(uint i) const
{
  assert(_mesh);
  assert(_mesh->ordered());

  // Initialize needed connectivity
  const uint D = _mesh->topology().dim();
  _mesh->init(2, D);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet, i);
}
//-----------------------------------------------------------------------------
Point Face::normal() const
{
  assert(_mesh);
  assert(_mesh->ordered());

  // Initialize needed connectivity
  const uint D = _mesh->topology().dim();
  _mesh->init(2, D);

  // Get cell to which face belong (first cell when there is more than one)
  const Cell cell(*_mesh, this->entities(D)[0]);

  // Get local index of facet with respect to the cell
  const uint local_facet = cell.index(*this);

  return cell.normal(local_facet);
}
//-----------------------------------------------------------------------------
