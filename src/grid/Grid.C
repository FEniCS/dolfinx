// Copyright (C) 2002 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <iostream.h>
#include <stdio.h>
#include <math.h>
#include <strings.h>

#include <Display.hh>
#include <utils.h>
#include <kw_constants.h>
#include <Input.hh>

#include "Grid.hh"
#include "Tetrahedron.hh"
#include "Triangle.hh"

//-----------------------------------------------------------------------------
Grid::Grid()
{
  no_nodes   = 0;
  no_cells   = 0;
  nodes    = 0;
  cells    = 0;
  celltype = CELL_NONE;
  mem      = sizeof(Grid);
  h        = 0.0;
}
//-----------------------------------------------------------------------------
Grid::~Grid()
{
  if ( nodes )
	 delete [] nodes;
  nodes = 0;
  
  if ( cells ){
	 for (int i=0;i<no_cells;i++)
		delete cells[i];
	 delete [] cells;
  }
  cells = 0;
}
//-----------------------------------------------------------------------------
void Grid::Init()
{
  ComputeNeighborInfo();
  ComputeSmallestDiameter();
}
//-----------------------------------------------------------------------------
void Grid::Clear()
{
  ClearNeighborInfo();
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

  switch ( celltype ) {
  case CELL_TRIANGLE:
	 sprintf(tmp,"triangles");
	 break;
  case CELL_TETRAHEDRON:
	 sprintf(tmp,"tetrahedrons");
	 break;
  default:
	 sprintf(tmp,"none");
  }
  
  display->Message(0,"Grid: %d nodes, %d cells (%s), %d bytes",
						 no_nodes,no_cells,tmp,mem);

}
//-----------------------------------------------------------------------------
void Grid::DisplayAll()
{
  Display();
  
  Node n;
  Point *p;
  Cell *c;
  
  display->Message(0,"");

  for (int i=0;i<no_nodes;i++){
	 n = nodes[i];
	 p = n.GetCoord();
	 printf("n = %d x = (%f,%f,%f) n-n = [ ",i,p->x,p->y,p->z);
	 for (int j=0;j<n.GetNoNodeNeighbors();j++)
		printf("%d ",n.GetNodeNeighbor(j));
	 printf("] n-c = [ ");
	 for (int j=0;j<n.GetNoCellNeighbors();j++)
		printf("%d ",n.GetCellNeighbor(j));
	 printf("]\n");
  }
  printf("\n");
  
  for (int i=0;i<no_cells;i++){
	 c = cells[i];
	 printf("c = %d n = ( ",i);
	 for (int j=0;j<c->GetSize();j++)
		printf("%d ",c->GetNode(j));
	 printf(") c-c = [ ");
	 for (int j=0;j<c->GetNoCellNeighbors();j++)
		printf("%d ",c->GetCellNeighbor(j));
	 printf("]\n");
  }
  printf("\n");

}
//-----------------------------------------------------------------------------
void Grid::Read(const char *filename)
{
  Input input(filename);

  int new_no_nodes = 0;
  int new_no_cells = 0;
  CellType new_celltype = CELL_NONE;

  // Get the number of nodes and elements
  input.ReadHeader(&new_no_nodes,&new_no_cells,&new_celltype);
  
  // Check data
  if ( new_no_nodes <= 0 )
	 display->Error("Number of nodes must be positive. (Reading grid file \"%s\".)",filename);
  if ( new_no_cells <= 0 )
	 display->Error("Number of cells must be positive. (Reading grid file \"%s\".)",filename);

  // Allocate memory for the nodes (also sets no_nodes)
  AllocNodes(new_no_nodes);
  
  // Read nodes
  input.ReadNodes(this,new_no_nodes);

  // Allocate memory for the cells (also sets no_cells)
  AllocCells(new_no_cells,new_celltype);

  // Read cells
  input.ReadCells(this,new_no_cells,new_celltype);

  // Compute maximum cell size
  ComputeMaximumCellSize();
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
//-----------------------------------------------------------------------------
void Grid::ComputeNeighborInfo()
{
  // Computes the following information:
  //
  // 1. All neighbor cells of a node: n-c
  // 2. All neighbor cells of a cell: c-c (including the cell itself)
  // 3. All neighbor nodes of a node: n-n (including the node itself)
  //
  // To save memory, the algorithm is not optimal w.r.t. speed.
  // The trick is to compute the connections in the correct order.
    
  int *tmp;
  int tmpsize, newtmpsize;

  // Reset all previous connections
  ClearNeighborInfo();

  // Prepare temporary storage
  tmp = new int[no_nodes];
  for (int i=0;i<no_nodes;i++)
	 tmp[i] = 0;
  
  display->Message(10,"Grid: Computing n-c connections.");
  
  // Start by computing the n-c connections. This is straightforward, since
  // we won't get any duplicate entries.
  for (int i=0;i<no_cells;i++)
	 cells[i]->CountCell(nodes);
  for (int i=0;i<no_nodes;i++)
	 nodes[i].AllocateForNeighborCells();
  for (int i=0;i<no_cells;i++)
	 cells[i]->AddCell(nodes,tmp,i);
  
  // Delete temporary storage
  delete [] tmp;
  
  display->Message(10,"Grid: Computing c-c connections.");
  
  // Now use this information to compute the c-c connections
  for (int i=0;i<no_cells;i++)
	 cells[i]->ComputeCellNeighbors(nodes,i);

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
}
//-----------------------------------------------------------------------------
