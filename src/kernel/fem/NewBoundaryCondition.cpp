// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewBoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewBoundaryCondition::NewBoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewBoundaryCondition::~NewBoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const dolfin::BoundaryValue NewBoundaryCondition::operator() (const Point& p)
{
  BoundaryValue value;
  return value;
}
//-----------------------------------------------------------------------------
