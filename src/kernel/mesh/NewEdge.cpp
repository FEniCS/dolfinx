// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells, 2006.
//
// First added:  2006-06-02
// Last changed: 2006-10-14

#include <dolfin/NewVertex.h>
#include <dolfin/NewEdge.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewPoint NewEdge::midpoint()
{
  uint* vertices = connections(0);
  dolfin_assert(vertices);

  const NewVertex v0(_mesh, vertices[0]);
  const NewVertex v1(_mesh, vertices[1]);
  
  const NewPoint p0 = v0.point();
  const NewPoint p1 = v1.point();

  NewPoint p(0.5*(p0.x() + p1.x()),
	     0.5*(p0.y() + p1.y()),
	     0.5*(p0.z() + p1.z()));

  return p;
}
//-----------------------------------------------------------------------------
