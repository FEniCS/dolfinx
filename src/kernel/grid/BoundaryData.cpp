// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <dolfin/BoundaryData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryData::BoundaryData(Grid* grid)
{
  this->grid = grid;
}
//-----------------------------------------------------------------------------
BoundaryData::~BoundaryData()
{
  clear();
}
//-----------------------------------------------------------------------------
void BoundaryData::clear()
{
  nodes.clear();
  edges.clear();
  faces.clear();
}
//-----------------------------------------------------------------------------
void BoundaryData::add(Node* node)
{
  nodes.add(node);
}
//-----------------------------------------------------------------------------
void BoundaryData::add(Edge* edge)
{
  edges.add(edge);
}
//-----------------------------------------------------------------------------
void BoundaryData::add(Face* face)
{
  faces.add(face);
}
//-----------------------------------------------------------------------------
bool BoundaryData::empty()
{
  return nodes.size() == 0 && edges.size() == 0 && faces.size() == 0;
}
//-----------------------------------------------------------------------------
int BoundaryData::noNodes() const
{
  return nodes.size();
}
//-----------------------------------------------------------------------------
int BoundaryData::noEdges() const
{
  return edges.size();
}
//-----------------------------------------------------------------------------
int BoundaryData::noFaces() const
{
  return faces.size();
}
//-----------------------------------------------------------------------------
