#include <dolfin/Grid.h>
#include <dolfin/Node.h>
#include <dolfin/GridIterators.h>
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
  for (GridNodeIterator n(*grid); !n.end(); ++n)
	 n->clear();
	 
  // Clear connectivity for cells
  for (GridCellIterator c(*grid); !c.end(); ++c)
	 c->clear();
}
//-----------------------------------------------------------------------------
void InitGrid::initNodeCell(Grid *grid)
{
  // Prepare temporary storage
  //tmp = new int[no_nodes];
  //for (int i=0;i<no_nodes;i++)
  //	 tmp[i] = 0;
  
  // Go through all cells and add the cell as a neighbour to all its nodes.
  // A couple of extra loops are needed to first count the number of
  // connections and then allocate memory for the connections.
  
  for (GridCellIterator c(*grid); !c.end(); ++c) {

	 
	 
	 
  }

  
  //for (int i=0;i<no_cells;i++)
  //	 cells[i]->CountCell(nodes);
  //for (int i=0;i<no_nodes;i++)
  //	 nodes[i].AllocateForNeighborCells();
  //for (int i=0;i<no_cells;i++)
  //	 cells[i]->AddCell(nodes,tmp,i);
  
  // Delete temporary storage
  // delete [] tmp;
  
}
//-----------------------------------------------------------------------------
void InitGrid::initCellCell(Grid *grid)
{
  /*
  // Now use this information to compute the c-c connections
  for (int i=0;i<no_cells;i++)
	 cells[i]->ComputeCellNeighbors(nodes,i);
  */
}
//-----------------------------------------------------------------------------
void InitGrid::initNodeNode(Grid *grid)
{
  /*
  // Prepare temporary storage
  tmpsize = nodes[0].GetMaxNodeNeighbors(cells);
  tmp = new int[tmpsize];
  
  display->Message(10,"Grid: Computing n-n connections.");
  
  // Now again use the n-c informaton to compute the n-n connections
  for (int i=0;i<no_nodes;i++){
	 newtmpsize = nodes[i].GetMaxNodeNeighbors(cells);
	 if ( newtmpsize > tmpsize ){
		tmpsize = newtmpsize;
		delete [] tmp;
		tmp = new int[tmpsize];
	 }
	 nodes[i].ComputeNodeNeighbors(cells,i,tmp);
  }
  
  // Clear temporary storage
  delete [] tmp;
  */
}
//-----------------------------------------------------------------------------

