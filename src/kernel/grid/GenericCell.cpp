// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/GenericCell.h>
#include <dolfin/Display.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
GenericCell::GenericCell()
{
  neighbor_cells = 0;
  nc = 0;

  _id = -1;
}
//-----------------------------------------------------------------------------
GenericCell::~GenericCell()
{
  Clear();
}
//-----------------------------------------------------------------------------
int GenericCell::id() const
{
  return _id;
}
//-----------------------------------------------------------------------------
void GenericCell::Clear()
{
  if ( neighbor_cells )
	 delete [] neighbor_cells;
  neighbor_cells = 0;

  nc = 0;
}
//-----------------------------------------------------------------------------
int GenericCell::setID(int id)
{
  return _id = id;
}
//-----------------------------------------------------------------------------
int GenericCell::GetNoGenericCellNeighbors()
{
  return ( nc );
}
//-----------------------------------------------------------------------------
int GenericCell::GetGenericCellNeighbor(int i)
{
  if ( (i<0) || (i>=nc) )
	 display->InternalError("GenericCell::GetGenericCellNeighbor()","Illegal index: %d",i);

  return ( neighbor_cells[i] );
}
//-----------------------------------------------------------------------------
int GenericCell::GetMaterial()
{
  return ( material );
}
//-----------------------------------------------------------------------------
