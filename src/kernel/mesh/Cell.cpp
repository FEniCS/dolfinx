// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-01-01
// Last changed: 2007-04-16

#include <dolfin/Cell.h>
#include <dolfin/Vertex.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Point Cell::midpoint()
{
  uint num_vertices = 0; 
  
  real x = 0.0;
  real y = 0.0;
  real z = 0.0;
  
  for (VertexIterator v(*this); !v.end(); ++v)
  {
    x += v->point().x();
    y += v->point().y();
    z += v->point().z();

    num_vertices++;
  }

  x /= real(num_vertices);
  y /= real(num_vertices);
  z /= real(num_vertices);

  Point p(x, y, z);
  return p;
}
//-----------------------------------------------------------------------------


