// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Grid.h>
#include <dolfin/BoundaryInit.h>
#include <dolfin/Boundary.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Boundary::Boundary(Grid& grid)
{
  this->grid = &grid;
  init();
}
//-----------------------------------------------------------------------------
Boundary::~Boundary()
{
  
}
//-----------------------------------------------------------------------------
void Boundary::init()
{
  if ( grid->bd.empty() )
    BoundaryInit::init(*grid);
}
//-----------------------------------------------------------------------------
void Boundary::clear()
{
  grid->bd.clear();
}
//-----------------------------------------------------------------------------
