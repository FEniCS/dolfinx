// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/Node.h>
#include <dolfin/Point.h>
#include <dolfin/Cell.h>
#include <dolfin/Triangle.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
int Triangle::noNodes() const
{
  return 3;
}
//-----------------------------------------------------------------------------
int Triangle::noEdges() const
{
  return 3;
}
//-----------------------------------------------------------------------------
int Triangle::noFaces() const
{
  return 1;
}
//-----------------------------------------------------------------------------
int Triangle::noBound() const
{
  return noEdges();
}
//-----------------------------------------------------------------------------
Cell::Type Triangle::type() const
{
  return Cell::TRIANGLE;
}
//-----------------------------------------------------------------------------
bool Triangle::neighbor(ShortList<Node *> &cn, Cell &cell) const
{
  // Two triangles are neighbors if they have a common edge or if they are
  // the same triangle, i.e. if they have 2 or 3 common nodes.

  if ( cell.type() != Cell::TRIANGLE )
	 return false;
  
  if ( !cell.c )
	 return false;

  int count = 0;
  for (int i = 0; i < 3; i++)
	 for (int j = 0; j < 3; j++)
		if ( cn(i) == cell.cn(j) )
		  count++;
  
  return count == 2 || count == 3;
}
//-----------------------------------------------------------------------------
