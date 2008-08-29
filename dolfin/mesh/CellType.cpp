// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-05
// Last changed: 2006-10-16

#include <dolfin/log/dolfin_log.h>
#include "PointCell.h"
#include "IntervalCell.h"
#include "TriangleCell.h"
#include "TetrahedronCell.h"
#include "Vertex.h"
#include "CellType.h"
#include "Cell.h"
#include "Point.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
CellType::CellType(Type cell_type, Type facet_type)
  : cell_type(cell_type), facet_type(facet_type)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellType::~CellType()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellType* CellType::create(Type type)
{
  switch ( type )
  {
  case point:
    return new PointCell();
  case interval:
    return new IntervalCell();
  case triangle:
    return new TriangleCell();
  case tetrahedron:
    return new TetrahedronCell();
  default:
    error("Unknown cell type: %d.", type);
  }
  
  return 0;
}
//-----------------------------------------------------------------------------
CellType* CellType::create(std::string type)
{
  return create(string2type(type));
}
//-----------------------------------------------------------------------------
CellType::Type CellType::string2type(std::string type)
{
  if ( type == "interval" )
    return interval;
  else if ( type == "triangle" )
    return triangle;
  else if ( type == "tetrahedron" )
    return tetrahedron;
  else
    error("Unknown cell type: \"%s\".", type.c_str());
  
  return interval;
}
//-----------------------------------------------------------------------------
bool CellType::intersects(MeshEntity& entity, Cell& cell) const
{
  for (VertexIterator v(entity); !v.end(); ++v)
  {
    Point p = v->point();

    if (intersects(cell, p))
      return true;
  }

  for (VertexIterator v(cell); !v.end(); ++v)
  {
    Point p = v->point();

    if(intersects(entity, p))
      return true;
  }

  return false;
}
//-----------------------------------------------------------------------------
std::string CellType::type2string(Type type)
{
  switch ( type )
  {
  case point:
    return "point";
  case interval:
    return "interval";
  case triangle:
    return "triangle";
  case tetrahedron:
    return "tetrahedron";
  default:
    error("Unknown cell type: %d.", type);
  }

  return "";
}
//-----------------------------------------------------------------------------
