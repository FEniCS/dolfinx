// Copyright (C) 2008 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2008-11-03
// Last changed: 2008-11-03

#include <dolfin/mesh/Cell.h>
#include "Data.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Data::Data()
  : x(0), t(0.0), _cell(0), _facet(-1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Data::Data(const Cell& cell, int facet)
  : x(0), t(0.0), _cell(&cell), _facet(facet)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Data::~Data()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const Cell& Data::cell() const
{
  if (!_cell)
    error("Current cell is unknown.");

  return *_cell;
}
//-----------------------------------------------------------------------------
dolfin::uint Data::facet() const
{
  if (_facet < 0)
    error("Current facet is unknown.");

  return static_cast<uint>(_facet);
}
//-----------------------------------------------------------------------------
Point Data::normal() const
{
  return cell().normal(facet());
}
//-----------------------------------------------------------------------------
bool Data::on_facet() const
{
  return _facet >= 0;
}
//-----------------------------------------------------------------------------
