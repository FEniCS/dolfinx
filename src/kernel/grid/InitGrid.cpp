#include <dolfin/Grid.h>
#include <dolfin/Node.h>
#include <dolfin/GenericCell.h>
#include <dolfin/InitGrid.h>

using namespace dolfin;

//-----------------------------------------------------------------------------
void InitGrid::init()
{
  // Reset all previous connections
  clear();

  // Init neighbor data
  initNeighbors();

  // Init boundary data
  initBoundary();
}
//-----------------------------------------------------------------------------
void InitGrid::clear()
{  
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
void InitGrid::initNeighbors()
{
  // Initialise neighbor information
  //
  //   n-c (cell neighbors of node)
  //   c-c (cell neighbors of cell)
  //   n-n (node neighbors of node)
  //
  // It is important that these are computed in the correct order.
  // Gives a nice and efficient algorithm that is O(n).
  
  // Compute n-c connections
  initNodeCell();
  
  // Compute c-c connections
  initCellCell();

  // Compute n-n connections
  initNodeNode();  
}
//-----------------------------------------------------------------------------
void InitGrid::initBoundary()
{
  // Go through all nodes and mark the nodes on the boundary. This is
  // done by checking if the neighbors of a node form a loop, using
  // Euler's theorem: A loop exists iff there are no odd nodes. A node
  // is odd if it has an odd number of arcs.

  int arcs;
  
  for (NodeIterator n1(grid); !n1.end(); ++n1) {

	 // First assume that the node is not on the boundary
	 n1->_boundary = -1;

	 // Count the number of odd nodes
	 for (NodeIterator n2(n1); !n2.end(); ++n2) {

		// Only check the neighbors
		if ( n2 == n1 )
		  continue;

		// Count arcs for n2
		arcs = 0;
		for (NodeIterator n3(n1); !n3.end(); ++n3)
		  if ( n2->neighbor(n3) && n3 != n1 && n3 != n2 )
			 arcs++;

		// Check if n2 is odd
		if ( (arcs % 2) != 0 ) {
		  n1->_boundary = 0;
		  break;
		}
		
	 }

  }
  
}
//-----------------------------------------------------------------------------
void InitGrid::initNodeCell()
{
  // Go through all cells and add the cell as a neighbour to all its nodes.
  // A couple of extra loops are needed to first count the number of
  // connections and then allocate memory for the connections.
  
  // Count the number of cells the node appears in
  for (CellIterator c(grid); !c.end(); ++c)
	 for (NodeIterator n(c); !n.end(); ++n)
		n->nc.setsize(n->nc.size()+1);
  
  // Allocate memory for the cell lists
  for (NodeIterator n(grid); !n.end(); ++n)
	 n->nc.init();
  
  // Add the cells to the cell lists
  for (CellIterator c(grid); !c.end(); ++c)
	 for (NodeIterator n(c); !n.end(); ++n)
		n->nc.add(c);  
}
//-----------------------------------------------------------------------------
void InitGrid::initCellCell()
{
  // Go through all cells and count the cell neighbors.

  for (CellIterator c1(grid); !c1.end(); ++c1) {

	 // Allocate for the maximum number of cell neighbors
	 c1->cc.init(c1->noBound());

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
void InitGrid::initNodeNode()
{
  // Go through all nodes and count the node neighbors.
  // This is done in four sweeps: count (overestimate), allocate, add, and
  // reallocate. Here is room for some optimisation: If for example the cell
  // is a triangle, then we will first allocate for 3n-3 node neighbors when
  // we really want only n.

  for (NodeIterator n1(grid); !n1.end(); ++n1) {

	 // Allocate for the maximum number of node neighbors
	 for (CellIterator c(n1); !c.end(); ++c)
		n1->nn.setsize( n1->nn.size() + c->noNodes() );
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
