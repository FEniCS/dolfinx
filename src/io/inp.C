#include "inp.h"

//-----------------------------------------------------------------------------
void inp_read_header(FILE *fp, int *no_nodes, int *no_cells,
							CellType *celltype)
{
  int d1,d2,d3,d4,d5;
  char string[DOLFIN_WORDLENGTH];

  // Rewind the file
  rewind(fp);
  
  // Read number of nodes and cells from first line
  fscanf(fp,"%d %d %d %d %d\n",no_nodes,no_cells,&d1,&d2,&d3);  

  // Skip nodes
  for (int i=0;i<(*no_nodes);i++)
	 skip_line(fp);

  // Read first line for elements and determine cell type
  fscanf(fp,"%d %d %s %d %d %d",&d1,&d2,string,&d3,&d4,&d5);
  if ( strcasecmp(string,"tri") == 0 )
	 *celltype = CELL_TRIANGLE;
  else if ( strcasecmp(string,"tet") == 0 )
	 *celltype = CELL_TETRAHEDRON;
  else
	 display->Error("Unknown cell type \"%s\" in grid file (seems to be an inp-file).",string);

  // Rewind the file and prepare for reading nodes
  rewind(fp);
  skip_line(fp);
}
//-----------------------------------------------------------------------------
void inp_read_nodes(FILE *fp, Grid *grid, int no_nodes)
{
  int n;
  float x,y,z;

  display->Message(0,"no_nodes = %d",no_nodes);
  
  // Read all nodes
  for (int i=0;i<no_nodes;i++){
	 fscanf(fp,"%d %f %f %f\n",&n,&x,&y,&z);
	 grid->GetNode(i)->SetCoord(x,y,z);
  }
}
//-----------------------------------------------------------------------------
void inp_read_cells(FILE *fp, Grid *grid, int no_cells, CellType celltype)
{
  int n, material, n1, n2, n3, n4;
  char string[DOLFIN_WORDLENGTH];

  if ( celltype == CELL_TRIANGLE )
	 for (int i=0;i<no_cells;i++){
		fscanf(fp,"%d %d %s %d %d %d\n",&n,&material,string,&n1,&n2,&n3);
		((Triangle *) grid->GetCell(i))->Set(n1-1,n2-1,n3-1,material);
	 }
  else if ( celltype == CELL_TETRAHEDRON )
	 for (int i=0;i<no_cells;i++){
		fscanf(fp,"%d %d %s %d %d %d %d\n",&n,&material,string,&n1,&n2,&n3,&n4);
		((Tetrahedron *) grid->GetCell(i))->Set(n1-1,n2-1,n3-1,n4-1,material);
	 }
}
//-----------------------------------------------------------------------------
