#include <dolfin/Grid.h>
#include <dolfin/Node.h>
#include <dolfin/GenericCell.h>
#include "InitGrid.h"

using namespace dolfin;

//-----------------------------------------------------------------------------
InitGrid::InitGrid(Grid *grid)
{
  int *tmp;
  int tmpsize, newtmpsize;

  // Reset all previous connections
  clear(grid);

  // Compute n-c connections
  initNodeCell(grid);
  
  // Compute c-c connections
  initCellCell(grid);

  // Compute n-n connections
  initNodeNode(grid);
}
//-----------------------------------------------------------------------------
void InitGrid::clear(Grid *grid)
{  
  cout << "Clearing connections" << endl;

  // Clear connectivity for nodes
  for (NodeIterator n(grid); !n.end(); ++n){
	 n->nn.clear();
	 n->nc.clear();
  }
	 
  // Clear connectivity for cells
  for (CellIterator c(grid); !c.end(); ++c)
	 c->cc.clear();
}
//-----------------------------------------------------------------------------
void InitGrid::initNodeCell(Grid *grid)
{
  // Go through all cells and add the cell as a neighbour to all its nodes.
  // A couple of extra loops are needed to first count the number of
  // connections and then allocate memory for the connections.
  
  // Count the number of cells the node appears in
  for (CellIterator c(grid); !c.end(); ++c)
	 for (NodeIterator n(c); !n.end(); ++n)
		n->nc.size()++;
  
  // Allocate memory for the cell lists
  for (NodeIterator n(grid); !n.end(); ++n)
	 n->nc.init();
  
  // Add the cells to the cell lists
  for (CellIterator c(grid); !c.end(); ++c)
	 for (NodeIterator n(c); !n.end(); ++n)
		n->nc.add(c);  
}
//-----------------------------------------------------------------------------
void InitGrid::initCellCell(Grid *grid)
{
  // Go through all cells and count the cell neighbors.

  for (CellIterator c1(grid); !c1.end(); ++c1) {

	 // Allocate for the maximum number of cell neighbors
	 c1->cc.init(c1->noBoundaries());

	 // Add all unique cell neighbors
	 for (NodeIterator n(c1); !n.end(); ++n)
		for (CellIterator c2(n); !c2.end(); ++c2)
		  if ( c1->neighbor(*c2) )
			 if ( !c1->cc.contains(c2) )
				c1->cc.add(c2);

	 // Reallocate
	 c1->cc.resize();

  }
  
}
//-----------------------------------------------------------------------------
void InitGrid::initNodeNode(Grid *grid)
{
  // Go through all nodes and count the node neighbors.
  // This is done in four sweeps: count (overestimate), allocate, add, and
  // reallocate. Here is room for some optimisation: If for example the cell
  // is a triangle, then we will first allocate for 3n-3 node neighbors when
  // we really want only n.

  for (NodeIterator n1(grid); !n1.end(); ++n1) {

	 // Allocate for the maximum number of node neighbors
	 for (CellIterator c(n1); !c.end(); ++c)
		n1->nn.size() += c->noNodes();
	 n1->nn.init();
	 
	 // Add all uniqe node neighbors
	 for (CellIterator c(n1); !c.end(); ++c)
		for (NodeIterator n2(c); !n2.end(); ++n2)
		  if ( !n1->nn.contains(n2) )
			 n1->nn.add(n2);

	 // Reallocate
	 n1->nn.resize();

  }
  
}
//-----------------------------------------------------------------------------

