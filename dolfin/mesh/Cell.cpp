// Copyright (C) 2006-2007 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Johan Hoffman 2006.
//
// First added:  2006-01-01
// Last changed: 2007-04-16

#include "Cell.h"
#include "Vertex.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
Point Cell::midpoint()
{
  uint num_vertices = 0; 
  
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  
  for (VertexIterator v(*this); !v.end(); ++v)
  {
    x += v->point().x();
    y += v->point().y();
    z += v->point().z();

    num_vertices++;
  }

  x /= double(num_vertices);
  y /= double(num_vertices);
  z /= double(num_vertices);

  Point p(x, y, z);
  return p;
}
//-----------------------------------------------------------------------------


