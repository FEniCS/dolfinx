// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-06

#include <dolfin/dolfin_log.h>
#include <dolfin/Interval.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint Interval::numEntities(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 2; // vertices
  case 1:
    return 1; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for interval.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
dolfin::uint Interval::numVertices(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for interval.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void Interval::createEntities(Array<Array<uint> >& entities,
			      uint dim, const uint vertices[])
{
  dolfin_error("Not implemented");
}
//-----------------------------------------------------------------------------
std::string Interval::description() const
{
  std::string s = "interval (simplex of topological dimension 1)";
  return s;
}
//-----------------------------------------------------------------------------
