// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Cell.h>
#include <dolfin/Map.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Map::Map()
{
  reset();
}
//-----------------------------------------------------------------------------
real Map::det() const
{
  return d;
}
//-----------------------------------------------------------------------------
real Map::bdet() const
{
  return bd;
}
//-----------------------------------------------------------------------------
int Map::boundary() const
{
  return _boundary;
}
//-----------------------------------------------------------------------------
const Cell& Map::cell() const 
{
  dolfin_assert(_cell);
  return *_cell;
}
//-----------------------------------------------------------------------------
void Map::update(const Edge& edge)
{
  dolfin_error("Non-matching update of map to boundary of cell.");
}
//-----------------------------------------------------------------------------
void Map::update(const Face& face)
{
  dolfin_error("Non-matching update of map to boundary of cell.");
}
//-----------------------------------------------------------------------------
real Map::ddx(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Map::ddy(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Map::ddz(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
real Map::ddt(real a) const
{
  return 0.0;
}
//-----------------------------------------------------------------------------
void Map::reset()
{
  f11 = f12 = f13 = 0.0;
  f21 = f22 = f23 = 0.0;
  f31 = f32 = f33 = 0.0;

  g11 = g12 = g13 = 0.0;
  g21 = g22 = g23 = 0.0;
  g31 = g32 = g33 = 0.0;
  
  d = 0.0;
  bd = 0.0;

  _boundary = -1;
  _cell = 0;
}
//-----------------------------------------------------------------------------
