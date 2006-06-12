// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-02
// Last changed: 2006-06-12

#include <dolfin/NewVertex.h>
#include <dolfin/NewEdge.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewPoint NewEdge::midpoint()
{
  // FIXME: Maybe this should be simplified?

  dolfin_assert(numConnections(0) > 0);
  uint* vertices = connections(0);
  dolfin_assert(vertices);

  NewVertex v0(_mesh, vertices[0]);
  NewVertex v1(_mesh, vertices[1]);

  NewPoint p0 = v0.point();
  NewPoint p1 = v1.point();

  NewPoint p(0.5*(v0.x() + v1.x()),
	     0.5*(v0.y() + v1.y()),
	     0.5*(v0.z() + v1.z()));

  return p;
}
//-----------------------------------------------------------------------------
