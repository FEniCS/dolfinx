// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream>
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
}
//-----------------------------------------------------------------------------
Grid::Grid(const char *filename) : initGrid(*this), refineGrid(*this)
{
  gd = 0;
  clear();

  // Read grid from file
  File file(filename);
  file >> *this;
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
Cell::Type Grid::type()
{
  // Warning: returns type of first cell
  CellIterator c(this);
  return c->type();
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
  return gd->createCell(type);
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(real x, real y, real z)
{
  return gd->createNode(x,y,z);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type, int n0, int n1, int n2)
{
  return gd->createCell(type,n0,n1,n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type, int n0, int n1, int n2, int n3)
{
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
std::ostream& dolfin::operator << (std::ostream& output, Grid& grid)
{
  output << "[ Grid with " << grid.noNodes() << " nodes and "
			<< grid.noCells() << " cells. ]";
  
  return output;
}
//-----------------------------------------------------------------------------
