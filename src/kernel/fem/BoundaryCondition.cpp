// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005
//
// First added:  2005-05-02
// Last changed: 2005-09-20

#include <dolfin/BoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition() : TimeDependent() 
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::~BoundaryCondition()
{
  // Do nothing
}
//-----------------------------------------------------------------------------
const dolfin::BoundaryValue BoundaryCondition::operator() (const Point& p)
{
  BoundaryValue value;
  return value;
}
//-----------------------------------------------------------------------------
const dolfin::BoundaryValue BoundaryCondition::operator() (const Point& p, 
							   uint i)
{
  BoundaryValue value;
  return value;
}
//-----------------------------------------------------------------------------
