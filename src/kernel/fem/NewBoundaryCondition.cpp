// Copyright (C) 2005 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/NewBoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
NewBoundaryCondition::NewBoundaryCondition(int no_comp) : no_comp(no_comp)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
NewBoundaryCondition::NewBoundaryCondition() 
{
  no_comp = 1;
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
const dolfin::BoundaryValue NewBoundaryCondition::operator() (const Point& p, 
							      const int i)
{
  BoundaryValue value;
  return value;
}
//-----------------------------------------------------------------------------
int NewBoundaryCondition::noComp() 
{
  return no_comp;
}
//-----------------------------------------------------------------------------
