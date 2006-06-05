// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-05

#include <dolfin/dolfin_log.h>
#include <dolfin/NewTetrahedron.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint NewTetrahedron::size(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 4; // vertices
  case 1:
    return 6; // edges
  case 2:
    return 4; // faces
  case 3:
    return 1; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
std::string NewTetrahedron::description() const
{
  std::string s = "tetrahedron (simplex of topological dimension 3)";
  return s;
}
//-----------------------------------------------------------------------------
