// Copyright (C) 2006 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2006-06-05
// Last changed: 2006-06-12

#include <dolfin/dolfin_log.h>
#include <dolfin/Interval.h>
#include <dolfin/NewTriangle.h>
#include <dolfin/NewTetrahedron.h>
#include <dolfin/CellType.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
CellType::CellType()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellType::~CellType()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
CellType* CellType::create(std::string type)
{
  if ( type == "interval" )
    return new Interval();
  else if ( type == "triangle" )
    return new NewTriangle();
  else if ( type == "tetrahedron" )
    return new NewTetrahedron();
  else
    dolfin_error1("Unknown cell type: \"%s\".", type.c_str());
  
  return 0;
}
//-----------------------------------------------------------------------------
CellType* CellType::create(Type type)
{
  switch ( type )
  {
  case interval:
    return new Interval();
  case triangle:
    return new NewTriangle();
  case tetrahedron:
    return new NewTetrahedron();
  default:
    dolfin_error("Unknown cell type.");
  }
  
  return 0;
}
//-----------------------------------------------------------------------------
