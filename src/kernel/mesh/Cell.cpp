// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-01-01
// Last changed: 2007-01-10

#include <dolfin/Cell.h>
#include <dolfin/Vertex.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Point Cell::midpoint()
{
  Point p;
  uint num_vertices = 0; 
  
  for (VertexIterator v(*this); !v.end(); ++v)
  {
    p.x() += v->point().x();
    p.y() += v->point().y();
    p.z() += v->point().z();

    num_vertices++;
  }

  p.x() /= real(num_vertices);
  p.y() /= real(num_vertices);
  p.z() /= real(num_vertices);

  return p;
}
//-----------------------------------------------------------------------------


