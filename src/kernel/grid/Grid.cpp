// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <math.h>
#include <strings.h>

#include <dolfin/dolfin_log.h>
#include <dolfin/utils.h>
#include <dolfin/constants.h>
#include <dolfin/Grid.h>
#include <dolfin/Node.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/NodeIterator.h>
#include <dolfin/File.h>
#include <dolfin/GridData.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
Grid::Grid() : initGrid(*this), refineGrid(*this)
{
  gd = 0;
  clear();

  rename("grid", "no description");
}
//-----------------------------------------------------------------------------
Grid::Grid(const char *filename) : initGrid(*this), refineGrid(*this)
{
  gd = 0;
  clear();
  
  // Read grid from file
  File file(filename);
  file >> *this;

  rename("grid", "no description");
}
//-----------------------------------------------------------------------------
Grid::~Grid()
{
  if ( gd )
	 delete gd;  
}
//-----------------------------------------------------------------------------
void Grid::clear()
{
  if ( gd )
	 delete gd;
  gd = new GridData();

  _type = TRIANGLES;
}
//-----------------------------------------------------------------------------
void Grid::operator = (const Grid& grid)
{
  init();  
  
  (*this->gd) = (*grid.gd);
}
//-----------------------------------------------------------------------------
void Grid::refine()
{
  refineGrid.refine();
}
//-----------------------------------------------------------------------------
int Grid::noNodes() const
{
  return gd->noNodes();
}
//-----------------------------------------------------------------------------
int Grid::noCells() const
{
  return gd->noCells();
}
//-----------------------------------------------------------------------------
Grid::Type Grid::type() const
{
  return _type;
}
//-----------------------------------------------------------------------------
void Grid::show()
{
  cout << "-------------------------------------------------------------------------------" << endl;
  cout << "Grid with " << noNodes() << " nodes and " << noCells() << " cells:" << endl;
  cout << endl;

  for (NodeIterator n(this); !n.end(); ++n)
	 cout << "  " << *n << endl;

  cout << endl;
  
  for (CellIterator c(this); !c.end(); ++c)
	 cout << "  " << *c << endl;
  
  cout << endl;
  
  cout << "-------------------------------------------------------------------------------" << endl;
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(int level)
{
  return gd->createNode(level);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int level, Cell::Type type)
{
  // Warning: grid type will be type of last added cell
  switch ( type ) {
  case Cell::TRIANGLE:
	 _type = TRIANGLES;
	 break;
  case Cell::TETRAHEDRON:
	 _type = TETRAHEDRONS;
	 break;
  default:
	 dolfin_error("Unknown cell type.");
  }
  
  return gd->createCell(level,type);
}
//-----------------------------------------------------------------------------
Edge* Grid::createEdge(int level)
{
  return gd->createEdge(level);
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(int level, Point p)
{
  return gd->createNode(level,p.x,p.y,p.z);
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(int level, real x, real y, real z)
{
  return gd->createNode(level,x,y,z);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int level, Cell::Type type, int n0, int n1, int n2)
{
  // Warning: grid type will be type of last added cell
  _type = TRIANGLES;
  
  return gd->createCell(level,type,n0,n1,n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int level, Cell::Type type, int n0, int n1, int n2, int n3)
{
  // Warning: grid type will be type of last added cell
  _type = TETRAHEDRONS;
  
  return gd->createCell(level,type,n0,n1,n2,n3);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2)
{
  // Warning: grid type will be type of last added cell
  _type = TRIANGLES;
  
  return gd->createCell(level,type,n0,n1,n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(int level, Cell::Type type, Node* n0, Node* n1, Node* n2, Node* n3)
{
  // Warning: grid type will be type of last added cell
  _type = TETRAHEDRONS;
  
  return gd->createCell(level,type,n0,n1,n2,n3);
}
//-----------------------------------------------------------------------------
Edge* Grid::createEdge(int level, int n0, int n1)
{
  return gd->createEdge(level,n0,n1);
}
//-----------------------------------------------------------------------------
Edge* Grid::createEdge(int level, Node* n0, Node* n1)
{
  return gd->createEdge(level,n0,n1);
}
//-----------------------------------------------------------------------------
Node* Grid::getNode(int id)
{
  return gd->getNode(id);
}
//-----------------------------------------------------------------------------
Cell* Grid::getCell(int id)
{
  return gd->getCell(id);
}
//-----------------------------------------------------------------------------
Edge* Grid::getEdge(int id)
{
  return gd->getEdge(id);
}
//-----------------------------------------------------------------------------
void Grid::init()
{
  initGrid.init();
}
//-----------------------------------------------------------------------------
// Additional operators
//-----------------------------------------------------------------------------
dolfin::LogStream& dolfin::operator<< (LogStream& stream, const Grid& grid)
{
  stream << "[ Grid with " << grid.noNodes() << " nodes and "
	 << grid.noCells() << " cells ";

  switch ( grid.type() ) {
  case Grid::TRIANGLES:
    stream << "(triangles) ]";
    break;
  case Grid::TETRAHEDRONS:
    stream << "(tetrahedrons) ]";
    break;
  default:
    stream << "(unknown type) ]";
  }

  return stream;
}
//-----------------------------------------------------------------------------
