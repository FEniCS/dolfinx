// Copyright (C) 2003-2006 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.
//
// First added:  2003
// Last changed: 2006-03-10

#include <dolfin/dolfin_log.h>
#include <dolfin/Vertex.h>
#include <dolfin/Edge.h>
#include <dolfin/Face.h>
#include <dolfin/BoundaryData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
BoundaryData::BoundaryData(Mesh& mesh)
{
  this->mesh = &mesh;
}
//-----------------------------------------------------------------------------
BoundaryData::~BoundaryData()
{
  clear();
}
//-----------------------------------------------------------------------------
void BoundaryData::clear()
{
  vertices.clear();
  edges.clear();
  faces.clear();
}
//-----------------------------------------------------------------------------
void BoundaryData::add(Vertex& vertex)
{
  //cout << "Adding vertex to boundary: " << vertex << endl;
  vertices.add(&vertex);
}
//-----------------------------------------------------------------------------
void BoundaryData::add(Edge& edge)
{
  //cout << "Adding edge to boundary: " << edge << endl;
  edges.add(&edge);
}
//-----------------------------------------------------------------------------
void BoundaryData::add(Face& face)
{
  //cout << "Adding face to boundary: " << face << endl;
  faces.add(&face);
}
//-----------------------------------------------------------------------------
bool BoundaryData::empty()
{
  return vertices.size() == 0 && edges.size() == 0 && faces.size() == 0;
}
//-----------------------------------------------------------------------------
int BoundaryData::numVertices() const
{
  return vertices.size();
}
//-----------------------------------------------------------------------------
int BoundaryData::numEdges() const
{
  return edges.size();
}
//-----------------------------------------------------------------------------
int BoundaryData::numFaces() const
{
  return faces.size();
}
//-----------------------------------------------------------------------------
void BoundaryData::setMesh(Mesh& mesh)
{
  this->mesh = &mesh;
}
//-----------------------------------------------------------------------------
