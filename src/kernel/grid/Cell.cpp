// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Cell.h>
#include <dolfin/Display.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Cell::Cell()
{
  neighbor_cells = 0;
  nc = 0;

  id = -1;
}
//-----------------------------------------------------------------------------
Cell::~Cell()
{
  Clear();
}
//-----------------------------------------------------------------------------
void Cell::Clear()
{
  if ( neighbor_cells )
	 delete [] neighbor_cells;
  neighbor_cells = 0;

  nc = 0;
}
//-----------------------------------------------------------------------------
int Cell::setID(int id)
{
  return this->id = id;
}
//-----------------------------------------------------------------------------
int Cell::GetNoCellNeighbors()
{
  return ( nc );
}
//-----------------------------------------------------------------------------
int Cell::GetCellNeighbor(int i)
{
  if ( (i<0) || (i>=nc) )
	 display->InternalError("Cell::GetCellNeighbor()","Illegal index: %d",i);

  return ( neighbor_cells[i] );
}
//-----------------------------------------------------------------------------
int Cell::GetMaterial()
{
  return ( material );
}
//-----------------------------------------------------------------------------
