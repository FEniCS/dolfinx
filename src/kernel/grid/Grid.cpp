// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream.h>
#include <stdio.h>
#include <math.h>
#include <strings.h>

#include <dolfin/Display.h>
#include <utils.h>
#include <kw_constants.h>

#include <dolfin/Grid.h>
#include <dolfin/Node.h>
#include <dolfin/Triangle.h>
#include <dolfin/Tetrahedron.h>
#include <dolfin/NodeIterator.h>
#include "GridData.h"
#include "InitGrid.h"


using namespace dolfin;

//-----------------------------------------------------------------------------
Grid::Grid()
{
  grid_data = 0;
  clear();
}
//-----------------------------------------------------------------------------
Grid::~Grid()
{
  if ( grid_data )
	 delete grid_data;  
}
//-----------------------------------------------------------------------------
void Grid::clear()
{
  if ( grid_data )
	 delete grid_data;
  grid_data = new GridData();

  no_nodes = 0;
  no_cells = 0;  
}
//-----------------------------------------------------------------------------
int Grid::noNodes()
{
  return no_nodes;
}
//-----------------------------------------------------------------------------
int Grid::noCells()
{
  return no_cells;
}
//-----------------------------------------------------------------------------
void Grid::show()
{
  cout << "-------------------------------------------------------------------------------" << endl;
  cout << "Grid with " << no_nodes << " nodes and " << no_cells << " cells:" << endl;
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
Node* Grid::createNode()
{
  no_nodes++;
  return grid_data->createNode();
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type)
{
  no_cells++;
  return grid_data->createCell(type);
}
//-----------------------------------------------------------------------------
Node* Grid::createNode(real x, real y, real z)
{
  no_nodes++;
  return grid_data->createNode(x,y,z);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type, int n0, int n1, int n2)
{
  no_cells++;
  return grid_data->createCell(type,n0,n1,n2);
}
//-----------------------------------------------------------------------------
Cell* Grid::createCell(Cell::Type type, int n0, int n1, int n2, int n3)
{
  no_cells++;
  return grid_data->createCell(type,n0,n1,n2,n3);
}
//-----------------------------------------------------------------------------
Node* Grid::getNode(int id)
{
  Node *node = grid_data->getNode(id);

  return node;
}
//-----------------------------------------------------------------------------
Cell* Grid::getCell(int id)
{
  Cell *cell = grid_data->getCell(id);

  return cell;
}
//-----------------------------------------------------------------------------
void Grid::init()
{
  InitGrid initGrid(this);
}
//-----------------------------------------------------------------------------
namespace dolfin {

  //---------------------------------------------------------------------------
  std::ostream& operator << (std::ostream& output, Grid& grid)
  {
	 int no_nodes = grid.noNodes();
	 int no_cells = grid.noCells();

	 output << "[ Grid with " << no_nodes << " nodes and "
			  << no_cells << " cells. ]";

	 return output;
  }
  //---------------------------------------------------------------------------

}
//-----------------------------------------------------------------------------

