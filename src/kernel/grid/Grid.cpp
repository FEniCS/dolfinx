// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <math.h>
#include <strings.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/utils.h>
#include <dolfin/constants.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/File.h>
#include <dolfin/GridHierarchy.h>
#include <dolfin/GridInit.h>
#include <dolfin/GridRefinement.h>
#include <dolfin/Grid.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Grid::Grid() : gd(this), bd(this), rd(this)
{
  rename("grid", "no description");
  clear();
}
//-----------------------------------------------------------------------------
Grid::Grid(const char* filename) : gd(this), bd(this), rd(this)
{
  rename("grid", "no description");
  clear();

  // Read grid from file
  File file(filename);
  file >> *this;
}
//-----------------------------------------------------------------------------
Grid::~Grid()
{
  clear();
}
//-----------------------------------------------------------------------------
void Grid::clear()
{
  gd.clear();
  bd.clear();
  rd.clear();
  _type = triangles;
  _parent = 0;
  _child = 0;
}
//-----------------------------------------------------------------------------
int Grid::noNodes() const
{
  return gd.noNodes();
}
//-----------------------------------------------------------------------------
int Grid::noCells() const
{
  return gd.noCells();
}
//-----------------------------------------------------------------------------
int Grid::noEdges() const
{
  return gd.noEdges();
}
//-----------------------------------------------------------------------------
int Grid::noFaces() const
{
  return gd.noFaces();
}
//-----------------------------------------------------------------------------
Grid::Type Grid::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
void Grid::mark(Cell* cell)
{
  rd.mark(cell);
}
//-----------------------------------------------------------------------------
void Grid::refine()
{
  // Check that this is the finest grid
  if ( _child )
    dolfin_error("Only the finest grid in a grid hierarchy can be refined.");

  // Create grid hierarchy
  GridHierarchy grids(*this);

  GridRefinement::refine(grids);
}
//-----------------------------------------------------------------------------
void Grid::show()
{
  cout << "---------------------------------------";
  cout << "----------------------------------------" << endl;

  cout << "Grid with " << noNodes() << " nodes and " 
       << noCells() << " cells:" << endl;
  cout << endl;

  for (NodeIterator n(this); !n.end(); ++n)
    cout << "  " << *n << endl;

  cout << endl;
  
  for (CellIterator c(this); !c.end(); ++c)
    cout << "  " << *c << endl;
  
  cout << endl;
  
  cout << "---------------------------------------";
  cout << "----------------------------------------" << endl;
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(Point p)
{
  return gd.createNode(p);
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(real x, real y, real z)
{
  return gd.createNode(x, y, z);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int n0, int n1, int n2)
{
  // Warning: grid type will be type of last added cell
  _type = triangles;
  
  return gd.createCell(n0, n1, n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int n0, int n1, int n2, int n3)
{
  // Warning: grid type will be type of last added cell
  _type = tetrahedrons;
  
  return gd.createCell(n0, n1, n2, n3);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Node* n0, Node* n1, Node* n2)
{
  // Warning: grid type will be type of last added cell
  _type = triangles;
  
  return gd.createCell(n0, n1, n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Node* n0, Node* n1, Node* n2, Node* n3)
{
  // Warning: grid type will be type of last added cell
  _type = tetrahedrons;
  
  return gd.createCell(n0, n1, n2, n3);
}
//-----------------------------------------------------------------------------
Edge* Grid::createEdge(int n0, int n1)
{
  return gd.createEdge(n0, n1);
}
//-----------------------------------------------------------------------------
Edge* Grid::createEdge(Node* n0, Node* n1)
{
  return gd.createEdge(n0, n1);
}
//-----------------------------------------------------------------------------
Face* Grid::createFace(int e0, int e1, int e2)
{
  return gd.createFace(e0, e1, e2);
}
//-----------------------------------------------------------------------------
Face* Grid::createFace(Edge* e0, Edge* e1, Edge* e2)
{
  return gd.createFace(e0, e1, e2);
}
//-----------------------------------------------------------------------------
Node* Grid::getNode(int id)
{
  return gd.getNode(id);
}
//-----------------------------------------------------------------------------
Cell* Grid::getCell(int id)
{
  return gd.getCell(id);
}
//-----------------------------------------------------------------------------
Edge* Grid::getEdge(int id)
{
  return gd.getEdge(id);
}
//-----------------------------------------------------------------------------
Face* Grid::getFace(int id)
{
  return gd.getFace(id);
}
//-----------------------------------------------------------------------------
bool Grid::hasEdge(Node* n0, Node* n1) const
{
  return gd.hasEdge(n0, n1);
}
//-----------------------------------------------------------------------------
void Grid::init()
{
  GridInit::init(*this);
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Grid& grid)
{
  stream << "[ Grid with " << grid.noNodes() << " nodes, "
	 << grid.noCells() << " cells ";

  switch ( grid.type() ) {
  case Grid::triangles:
    stream << "(triangles)";
    break;
  case Grid::tetrahedrons:
    stream << "(tetrahedrons)";
    break;
  default:
    stream << "(unknown type)";
  }

  stream << ", and " << grid.noEdges() << " edges ]";

  return stream;
}
//-----------------------------------------------------------------------------
