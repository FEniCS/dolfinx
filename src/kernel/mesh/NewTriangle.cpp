// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-05

#include <dolfin/dolfin_log.h>
#include <dolfin/NewTriangle.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint NewTriangle::size(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 3; // vertices
  case 1:
    return 3; // edges
  case 2:
    return 1; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::string NewTriangle::description() const
{
  std::string s = "triangle (simplex of topological dimension 2)";
  return s;
}
//-----------------------------------------------------------------------------
