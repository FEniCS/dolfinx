// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-06

#include <dolfin/dolfin_log.h>
#include <dolfin/NewTriangle.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
dolfin::uint NewTriangle::numEntities(uint dim) const
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
dolfin::uint NewTriangle::numVertices(uint dim) const
{
  switch ( dim )
  {
  case 0:
    return 1; // vertices
  case 1:
    return 2; // edges
  case 2:
    return 3; // cells
  default:
    dolfin_error1("Illegal topological dimension %d for tetrahedron.", dim);
  }

  return 0;
}
//-----------------------------------------------------------------------------
void NewTriangle::createEntities(Array<Array<uint> >& entities,
				 uint dim, uint vertices[])
{
  // We only need to know how to create edges
  if ( dim != 1 )
    dolfin_error1("Don't know how to create entities of topological dimension %d.", dim);


  /*
  dolfin_assert(entities.size() == 3);

  // Create the three edges
  entities[0] = {vertices[0], vertices[2]};
  entities[1] = {vertices[2], vertices[0]};
  entities[2] = {vertices[0], vertices[1]};
  */
}
//-----------------------------------------------------------------------------
std::string NewTriangle::description() const
{
  std::string s = "triangle (simplex of topological dimension 2)";
  return s;
}
//-----------------------------------------------------------------------------
