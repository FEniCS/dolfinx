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

  grid_data = new GridData();
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

  
  nodes    = 0;
  cells    = 0;
  mem      = sizeof(Grid);
  h        = 0.0;
  
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


//-----------------------------------------------------------------------------
void Grid::Init()
{
  InitGrid initGrid(this);

  //  ComputeNeighborInfo();
  // ComputeSmallestDiameter();

}
//-----------------------------------------------------------------------------
void Grid::Clear()
{
  // ClearNeighborInfo();
}
//-----------------------------------------------------------------------------
int Grid::GetNoNodes()
{
  return ( no_nodes );
}
//-----------------------------------------------------------------------------
int Grid::GetNoCells()
{
  return ( no_cells );
}
//-----------------------------------------------------------------------------
int Grid::GetMaxCellSize()
{
  return ( maxcellsize );
}
//-----------------------------------------------------------------------------
bool Grid::AllCellsEqual()
{
  // No support for hybrid grids yet. All cells are the same.

  return true;
}
//-----------------------------------------------------------------------------
real Grid::GetSmallestDiameter()
{
  if ( h == 0.0 )
	 display->Error("Diameter of smallest cell has not been computed.");

  return h;
}
//-----------------------------------------------------------------------------
Node* Grid::GetNode(int node)
{
   if ( (node < 0) || (node >= no_nodes) )
  	 display->InternalError("Grid::GetNode","Illegal node number: %d",node);

	return ( nodes+node );
}
//-----------------------------------------------------------------------------
Cell* Grid::GetCell(int cell)
{
  if ( (cell < 0) || (cell >= no_cells) ){
    display->InternalError("Grid::GetCell","Illegal cell number.");
  }

  return ( cells[cell] );
}
//-----------------------------------------------------------------------------
void Grid::Display()
{
  char tmp[DOLFIN_WORDLENGTH];

  //  switch ( celltype ) {
  //case CELL_TRIANGLE:
  // sprintf(tmp,"triangles");
  //	 break;
  //case CELL_TETRAHEDRON:
  //	 sprintf(tmp,"tetrahedrons");
  //	 break;
  //default:
  //	 sprintf(tmp,"none");
  // }
  
  //display->Message(0,"Grid: %d nodes, %d cells (%s), %d bytes",
  //						 no_nodes,no_cells,tmp,mem);

}
//-----------------------------------------------------------------------------
void Grid::DisplayAll()
{

}
//-----------------------------------------------------------------------------
void Grid::Read(const char *filename)
{
  /*
  Input input(filename);

  int new_no_nodes = 0;
  int new_no_cells = 0;
  //  CellType new_celltype = CELL_NONE;

  // Get the number of nodes and elements
  //  input.ReadHeader(&new_no_nodes,&new_no_cells,&new_celltype);
  
  // Check data
  if ( new_no_nodes <= 0 )
	 display->Error("Number of nodes must be positive. (Reading grid file \"%s\".)",filename);
  if ( new_no_cells <= 0 )
	 display->Error("Number of cells must be positive. (Reading grid file \"%s\".)",filename);

  */
	 
  // Allocate memory for the nodes (also sets no_nodes)
  // AllocNodes(new_no_nodes);
  
  // Read nodes
  //input.ReadNodes(this,new_no_nodes);

  // Allocate memory for the cells (also sets no_cells)
  //AllocCells(new_no_cells,new_celltype);

  // Read cells
  //input.ReadCells(this,new_no_cells,new_celltype);

  // Compute maximum cell size
  //  ComputeMaximumCellSize();
}
//-----------------------------------------------------------------------------
void Grid::Write(const char *filename)
{
  display->Message(10,"Writing grid file: \"%s\"",filename);

  // Try to open the file for writing
  FILE *fp = fopen(filename,"w");
  if ( !fp )
	 display->Error("Unable to write grid file: %s.",filename);
  
  if ( suffix(filename,".inp") ) {
	 display->Message(10,"Seems to be an .inp-file (from suffix).");
    WriteINP(fp);
  }
  else{
	 fclose(fp);
	 display->Error("Unknown file format for grid.");
  }
  
  // Close the file
  fclose(fp);
}
//-----------------------------------------------------------------------------







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
void Grid::WriteINP(FILE *fp)
{
  Point *p;
  Triangle *tri;
  Tetrahedron *tet;
  
  // Write first line
  fprintf(fp,"%d %d 0 0 0\n",no_nodes,no_cells);
  
  // Write node position
  for (int i=0;i<no_nodes;i++){
	 p = &nodes[i].p;
	 fprintf(fp,"%d %f %f %f\n",i+1,p->x,p->y,p->z);
  }

  // Write cells
  /*
  switch ( celltype ){
  case CELL_TRIANGLE:
	 for (int i=0;i<no_cells;i++){
		tri = (Triangle *) cells[i];
		fprintf(fp,"%d %d tri %d %d %d\n",i+1,
				  tri->material,
				  tri->nodes[0]+1,tri->nodes[1]+1,tri->nodes[2]+1);
	 }
	 break;
  case CELL_TETRAHEDRON:
	 for (int i=0;i<no_cells;i++){
		tet = (Tetrahedron *) cells[i];
		fprintf(fp,"%d %d tet %d %d %d %d\n",i+1,
				  tri->material,
				  tet->nodes[0],tet->nodes[1]+1,tet->nodes[2]+1,tet->nodes[3]+1);
	 }
	 break;
  default:
	 display->InternalError("Grid::WriteINP()","Unknown cell type: %d",
									celltype);
  }
  */
  
}
//-----------------------------------------------------------------------------
void Grid::AllocNodes(int newsize)
{
  // Delete old nodes if any
  if ( nodes )
	 delete [] nodes;
  
  // Keep track of allocated size
  mem -= no_nodes*sizeof(Node);
  
  // Allocate new nodes
  nodes = new Node[newsize];
  
  // Check that we got the memory
  if ( !nodes )
	 display->Error("Out of memory. Unable to allocate new memory for Grid.");
  
  // Save the number of nodes
  no_nodes = newsize;
  
  // Keep track of allocated size
  mem += no_nodes*sizeof(Node);
}
//-----------------------------------------------------------------------------
/*
void Grid::AllocCells(int newsize, CellType newtype)
{
  // Delete old cells
  if ( cells ){

	 for (int i=0;i<no_cells;i++)
		delete cells[i];
	 delete [] cells;
	 cells = 0;
	 
	 switch ( celltype ){
	 case CELL_TRIANGLE:
		mem -= no_cells*sizeof(Triangle);
		break;
	 case CELL_TETRAHEDRON:
		mem -= no_cells*sizeof(Tetrahedron);
		break;
	 default:
		display->InternalError("Grid::AllocCells()","Unknown cell type: %d",celltype);
	 }
	 
  }
  
  // Save data
  no_cells = newsize;
  celltype = newtype;

  // Allocate memory for cell pointers
  cells = new (Cell *)[no_cells];
  mem += no_cells*sizeof(Cell *);
  
  // Allocate memory for the cells themselves
  switch ( celltype ){
  case CELL_TRIANGLE:
	 for (int i=0;i<no_cells;i++)
		cells[i] = new Triangle;
	 mem  += no_cells*sizeof(Triangle);
	 break;
  case CELL_TETRAHEDRON:
	 for (int i=0;i<no_cells;i++)
		cells[i] = new Tetrahedron;
	 mem  += no_cells*sizeof(Triangle);
	 break;
  default:
	 display->InternalError("Grid::AllocCells()","Unknown cell type: %d",celltype);
  }

  // Check that we got the memory
  if ( !cells )
	 display->Error("Out of memory. Unable to allocate new memory for Grid.");

}
//-----------------------------------------------------------------------------
void Grid::ClearNeighborInfo()
{
  // Clear nodes
  for (int i=0;i<no_nodes;i++)
	 nodes[i].Clear();

  // Clear cells
  for (int i=0;i<no_cells;i++)
	 cells[i]->Clear();
  
  // FIXME: Add memory tracking

}
//-----------------------------------------------------------------------------
void Grid::ComputeMaximumCellSize()
{
  int maxsize = 0;
  int n;
  
  for (int i=0;i<no_cells;i++)
	 if ( (n=cells[i]->GetSize()) > maxsize )
		maxsize = n;
}
//-----------------------------------------------------------------------------
void Grid::ComputeSmallestDiameter()
{
  real hh;
  
  h = 2.0 * cells[0]->ComputeCircumRadius(this);
  
  for (int i=1;i<no_cells;i++)
	 if ( (hh = 2.0 * cells[i]->ComputeCircumRadius(this)) < h )
		h = hh;
}
*/
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

