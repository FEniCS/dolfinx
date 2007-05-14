// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU LGPL Version 2.1.
//
// First added:  2006-06-05
// Last changed: 2006-10-16

#include <dolfin/dolfin_log.h>
#include <dolfin/Interval.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/Vertex.h>
#include <dolfin/CellType.h>
#include <dolfin/Cell.h>
#include <dolfin/Point.h>

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
  case interval:
    return new Interval();
  case triangle:
    return new Triangle();
  case tetrahedron:
    return new Tetrahedron();
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
bool CellType::intersects(MeshEntity& entity, Cell& c) const
{
  for(VertexIterator vi(entity); !vi.end(); ++vi)
  {
    Point p = vi->point();

    if(intersects(c, p))
      return true;
  }

  for(VertexIterator vi(c); !vi.end(); ++vi)
  {
    Point p = vi->point();

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
