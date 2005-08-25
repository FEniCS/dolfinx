// Copyright (C) 2005 Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// Modified by Garth N. Wells 2005
//
// First added:  2005-05-02
// Last changed: 2005-08-23

#include <dolfin/BoundaryCondition.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition(uint num_components)
  : num_components(num_components)
{
  // Do nothing
}
//-----------------------------------------------------------------------------
BoundaryCondition::BoundaryCondition() 
{
  num_components = 1;
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
							   const int i)
{
  BoundaryValue value;
  return value;
}
//-----------------------------------------------------------------------------
dolfin::uint BoundaryCondition::numComponents() const
{
  return num_components;
 }
//-----------------------------------------------------------------------------
real BoundaryCondition::time() const
{
  return t;
}
//-----------------------------------------------------------------------------
void BoundaryCondition::set(real time)
{
  t = time;
}
//-----------------------------------------------------------------------------
 
