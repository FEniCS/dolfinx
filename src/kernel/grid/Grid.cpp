// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <math.h>
#include <strings.h>

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
  std::cout << "-------------------------------------------------------------------------------" << std::endl;
  std::cout << "Grid with " << noNodes() << " nodes and " << noCells() << " cells:" << std::endl;
  std::cout << std::endl;

  for (NodeIterator n(this); !n.end(); ++n)
	 std::cout << "  " << *n << std::endl;

  std::cout << std::endl;
  
  for (CellIterator c(this); !c.end(); ++c)
	 std::cout << "  " << *c << std::endl;
  
  std::cout << std::endl;
  
  std::cout << "-------------------------------------------------------------------------------" << std::endl;
}
//-----------------------------------------------------------------------------
Node* Grid::createNode()
{
  return gd->createNode();
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type)
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
	 std::cout << "Warning: unknown cell type." << std::endl;
  }
  
  return gd->createCell(type);
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(Point p)
{
  return gd->createNode(p.x,p.y,p.z);
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(real x, real y, real z)
{
  return gd->createNode(x,y,z);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type, int n0, int n1, int n2)
{
  // Warning: grid type will be type of last added cell
  _type = TRIANGLES;
  
  return gd->createCell(type,n0,n1,n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type, int n0, int n1, int n2, int n3)
{
  // Warning: grid type will be type of last added cell
  _type = TETRAHEDRONS;
  
  return gd->createCell(type,n0,n1,n2,n3);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type, Node* n0, Node* n1, Node* n2)
{
  // Warning: grid type will be type of last added cell
  _type = TRIANGLES;
  
  return gd->createCell(type,n0,n1,n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type, Node* n0, Node* n1, Node* n2, Node* n3)
{
  // Warning: grid type will be type of last added cell
  _type = TETRAHEDRONS;
  
  return gd->createCell(type,n0,n1,n2,n3);
}
//-----------------------------------------------------------------------------
Node* Grid::getNode(int id)
{
  Node *node = gd->getNode(id);

  return node;
}
//-----------------------------------------------------------------------------
Cell* Grid::getCell(int id)
{
  Cell *cell = gd->getCell(id);

  return cell;
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
			<< grid.noCells() << " cells. ]";

  return stream;
}
//-----------------------------------------------------------------------------
