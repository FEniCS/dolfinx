// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/dolfin_log.h>
#include <dolfin/Cell.h>
#include <dolfin/Point.h>
#include <dolfin/Node.h>
#include <dolfin/P1IntMap.h>
#include <cmath>

using namespace dolfin;

//-----------------------------------------------------------------------------
P1IntMap::P1IntMap() : Map()
{
  // Set dimension
  dim = 1;
}
//-----------------------------------------------------------------------------
void P1IntMap::update(const Cell& cell)
{
  dolfin_error("Intervals (1D) not yet implemented in DOLFIN.");

  // Check that cell type is correct
  //if ( cell.type() != Cell::interval )
  //  dolfin_error("Wrong cell type for map (must be an interval).");
  
  cell_ = &cell;

  // Reset values
  reset();
  
  // Get coordinates
  NodeIterator n(cell);
  Point p0 = n->coord(); ++n;
  Point p1 = n->coord(); 

  // Set values for Jacobian
  f11 = p1.x - p0.x; 
  
  // Compute determinant
  d = f11;
  
  // Check determinant
  if ( fabs(d) < DOLFIN_EPS )
    dolfin_error("Map from reference element is singular.");
  
  // Compute inverse
  g11 = 1 / d;
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1IntMap::ddx
(const FunctionSpace::ShapeFunction& v) const
{
  return g11*v.ddX();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1IntMap::ddy
(const FunctionSpace::ShapeFunction& v) const
{
  return v.ddY();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1IntMap::ddz
(const FunctionSpace::ShapeFunction& v) const
{
  return v.ddZ();
}
//-----------------------------------------------------------------------------
const FunctionSpace::ElementFunction P1IntMap::ddt
(const FunctionSpace::ShapeFunction& v) const
{
  return v.ddT();
}
//-----------------------------------------------------------------------------
