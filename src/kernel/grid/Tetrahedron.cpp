// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/Tetrahedron.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
int Tetrahedron::noNodes() const
{
  return 4;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noEdges() const
{
  return 6;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noFaces() const
{
  return 4;
}
//-----------------------------------------------------------------------------
int Tetrahedron::noBound() const
{
  return noFaces();
}
//-----------------------------------------------------------------------------
Cell::Type Tetrahedron::type() const
{
  return Cell::TETRAHEDRON;
}
//-----------------------------------------------------------------------------
bool Tetrahedron::neighbor(ShortList<Node *> &cn, Cell &cell) const
{
  // Two tetrahedrons are neighbors if they have a common face or if they are
  // the same tetrahedron, i.e. if they have 3 or 4 common nodes.
  
  if ( cell.type() != Cell::TETRAHEDRON )
	 return false;
  
  if ( !cell.c )
	 return false;

  int count = 0;
  for (int i = 0; i < 4; i++)
	 for (int j = 0; j < 4; j++)
		if ( cn(i) == cell.cn(j) )
		  count++;
  
  return count == 3 || count == 4;
}
//-----------------------------------------------------------------------------
