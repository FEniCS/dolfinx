// Copyright (C) 2009 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2009-03-11
// Last changed: 2009-10-11

#include <dolfin/mesh/Cell.h>
#include "Data.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Data::Data()
  : x(0), values(0), _dolfin_cell(0), _ufc_cell(0), _facet(-1)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
Data::Data(uint geometric_dim, uint value_size)
  : x(geometric_dim), values(value_size), _dolfin_cell(0), _ufc_cell(0), 
    _facet(-1)
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
  if (!_dolfin_cell)
    error("Current cell is unknown.");

  return *_dolfin_cell;
}
//-----------------------------------------------------------------------------
const ufc::cell& Data::ufc_cell() const
{
  if (!_ufc_cell)
    error("Current UFC cell is unknown.");

  return *_ufc_cell;
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
dolfin::uint Data::geometric_dimension() const
{
  assert(_dolfin_cell);
  return _dolfin_cell->mesh().geometry().dim();
}
//-----------------------------------------------------------------------------
bool Data::on_facet() const
{
  return _facet >= 0;
}
//-----------------------------------------------------------------------------
void Data::set_geometric_dim(uint dim)
{
  x.resize(dim);
}
//-----------------------------------------------------------------------------
void Data::set_value_size(uint dim)
{
  values.resize(dim);
}
//-----------------------------------------------------------------------------
void Data::set(const Cell& dolfin_cell,
                  const ufc::cell& ufc_cell,
                  int local_facet)
{
  _dolfin_cell = &dolfin_cell;
  _ufc_cell = &ufc_cell;
  _facet = local_facet;
}
//-----------------------------------------------------------------------------
void Data::set(const ufc::cell& ufc_cell, const double* x)
{
  _ufc_cell = &ufc_cell;
  this->x.resize(ufc_cell.geometric_dimension);
  this->x.assign(x, x + this->x.size()); 
}
//-----------------------------------------------------------------------------
void Data::clear()
{
  x.resize(0);
  _dolfin_cell = 0;
  _ufc_cell = 0;
  _facet = -1;
}
//-----------------------------------------------------------------------------
