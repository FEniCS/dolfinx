// (c) 2002 Johan Hoffman & Anders Logg, Chalmers Finite Element Center.
// Licensed under the GNU GPL Version 2.
//
// The following people have contributed to the GiD output code:
//
//   Rasmus Hemph       (PDE project course 2001/2002)
//   Alexandra Krusper  (PDE project course 2001/2002)
//   Walter Villanueva  (PDE project course 2001/2002)

#include "gid.h"

//-----------------------------------------------------------------------------
void gid_read_header(FILE *fp, int *no_nodes, int *no_cells,
							CellType *celltype)
{
  int i1, i2;
  char s1[DOLFIN_WORDLENGTH];
  char s2[DOLFIN_WORDLENGTH];
  char s3[DOLFIN_WORDLENGTH];
  char s4[DOLFIN_WORDLENGTH];
  char s5[DOLFIN_WORDLENGTH];

  // Rewind the file to the beginning
  rewind(fp);

  // Read first line to determine element type
  fscanf(fp,"%s %s %d %s %s %s %d\n",s1,s2,&i1,s3,s4,s5,&i2);
  if ( strcasecmp(s4,"Triangle") == 0 )
	 *celltype = CELL_TRIANGLE;
  else if ( strcasecmp(s4,"Tetrahedra") == 0 )
	 *celltype = CELL_TETRAHEDRON;
  else
	 display->Error("Unknown cell type \"%s\" in grid file (seems to be a GiD-file).",s4);
  
  // Warning: i1 is the dimension but we assume this is 3 anyway.
  
  // Find the "Coordinates" keyword
  while ( !keyword_in_line(fp,"Coordinates") );
  
  // Count the number of nodes
  *no_nodes = 0;
  while ( !keyword_in_line(fp,"end coordinates") )
	 *no_nodes += 1;

  // Find the "Elements" keyword
  while ( !keyword_in_line(fp,"Elements") );

  // Count the number of cells
  *no_cells = 0;
  while ( !keyword_in_line(fp,"end elements") )
	 *no_cells += 1;  
}
//-----------------------------------------------------------------------------
void gid_read_nodes(FILE *fp, Grid *grid, int no_nodes)
{
  // Rewind the file
  rewind(fp);

  // Find the "Coordinates" keyword
  while ( !keyword_in_line(fp,"Coordinates") );

  // Read all nodes
  int n;
  float x,y,z;

  for (int i=0;i<no_nodes;i++){
	 fscanf(fp,"%d %f %f %f\n",&n,&x,&y,&z);
	 grid->GetNode(i)->SetNodeNo(n-1);
	 grid->GetNode(i)->SetCoord(x,y,z);
  }
}
//-----------------------------------------------------------------------------
void gid_read_cells(FILE *fp, Grid *grid, int no_cells, CellType celltype)
{
  // Find the "Elements" keyword
  while ( !keyword_in_line(fp,"Elements") );
  
  // Read all cells
  int n,n1,n2,n3,n4;

  if ( celltype == CELL_TRIANGLE )
	 for (int i=0;i<no_cells;i++){
		fscanf(fp,"%d %d %d %d\n",&n,&n1,&n2,&n3);
		((Triangle *) grid->GetCell(i))->
		  Set(grid->GetNode(n1-1),grid->GetNode(n2-1),grid->GetNode(n3-1),0);
	 }
  else if ( celltype == CELL_TETRAHEDRON )
	 for (int i=0;i<no_cells;i++){
		fscanf(fp,"%d %d %d %d %d\n",&n,&n1,&n2,&n3,&n4);
		((Tetrahedron *) grid->GetCell(i))->
		  Set(grid->GetNode(n1-1),grid->GetNode(n2-1),
		      grid->GetNode(n3-1),grid->GetNode(n4-1),0);
	 }
}
//-----------------------------------------------------------------------------
void gid_write_header(FILE *fp, DataInfo *datainfo, SysInfo *sysinfo)
{
  fprintf(fp,"# Output from DOLFIN version %s.\n",DOLFIN_VERSION);
  fprintf(fp,"# Format is intended for use with GiD.\n");
  fprintf(fp,"#\n");
  fprintf(fp,"# Saved by %s at %s\n",sysinfo->user,sysinfo->time);
  fprintf(fp,"# on %s (%s) running %s version %s.\n",
			 sysinfo->host,sysinfo->mach,sysinfo->name,sysinfo->vers);
  fprintf(fp,"#\n");
  fprintf(fp,"#    %s\n",datainfo->Description());
  fprintf(fp,"#\n");
  fprintf(fp,"# Data series in this file:\n");
  fprintf(fp,"#\n");
  for (int i=0;i<datainfo->Size();i++)
	 fprintf(fp,"#    %s: %s (dimension = %d)\n",
				datainfo->Label(i),
				datainfo->Name(i),
				datainfo->Dim(i));
  fprintf(fp,"\n");


}
//-----------------------------------------------------------------------------
void gid_write_grid(FILE *fp, Grid *grid)
{
  Point *p;
  Cell *c;;
  
  // Write grid header
  if ( grid->GetCell(0)->GetSize() == 3 )
    fprintf(fp, "MESH \"DOLFIN\" dimension 2 Elemtype Triangle Nnode 3\n");
  else
    fprintf(fp, "MESH \"DOLFIN\" dimension 3 Elemtype Tetrahedra Nnode 4\n");

  // Write coordinates
  fprintf(fp, "Coordinates \n");
  for (int i=0;i<grid->GetNoNodes();i++){
    p = grid->GetNode(i)->GetCoord();
    fprintf(fp,"%i %f %f %f\n",i+1, p->x, p->y, p->z);
  }
  fprintf(fp, "end coordinates\n");
 
  // Write cells
  fprintf(fp,"Elements\n");  
  for (int i=0;i<grid->GetNoCells();i++){
    c = grid->GetCell(i);
    fprintf(fp,"%i",i+1);
    for (int j=0;j<c->GetSize();j++)
		  fprintf(fp," %d",c->GetNode(j)->GetNodeNo()+1);
	 fprintf(fp,"\n");
  }
  fprintf(fp,"end elements\n");
}
//-----------------------------------------------------------------------------
void gid_write_field(FILE *fp, Grid *grid, DataInfo *datainfo, Vector **u,
							real t, int frame, int no_vectors)
{
  int no_nodes = grid->GetNoNodes();
  int offset   = 0;

  // Compute total data size
  int datasize = 0;
  for (int i=0;i<datainfo->Size();i++)
	 datasize += datainfo->Dim(i);
  
  // Write all fields
  for (int i=0;i<datainfo->Size();i++){

	 // Write header for data
	 if ( datainfo->Dim(i) > 1 ){
		fprintf(fp,"Result \"%s\" \"DOLFIN results\" %d Vector OnNodes\n",datainfo->Label(i),frame+1);
		//fprintf(fp,"ComponentNames");
		//for (int j=0;j<datainfo->Dim(i);j++)
		//  fprintf(fp," \"%s\"",datainfo->Name(i),j+1);
		//fprintf(fp,"\n");
	 }
	 else{
		fprintf(fp,"Result \"%s\" \"DOLFIN results\" %d Scalar OnNodes\n",datainfo->Label(i),frame+1);
		//fprintf(fp,"ComponentNames %s\n",datainfo->Name(i));
	 }
		
	 // Write values
	 fprintf(fp,"%s step %d\n",datainfo->Name(i),frame+1);
	 	 
	 if ( no_vectors == 1 ){
		
		for (int j=0;j<no_nodes;j++){
		  fprintf(fp,"%d",j+1);
		  for (int k=0;k<datainfo->Dim(i);k++)
			 fprintf(fp," %f", float( u[0]->Get(j*datasize+offset+k) ) );
		  fprintf(fp,"\n");
		}
		
		offset += datainfo->Dim(i);
		  
	 }
	 else{

		for (int j=0;j<u[i]->Size();j++){
		  fprintf(fp,"%d",j+1);
		  for (int k=0;k<datainfo->Dim(i);k++)
			 fprintf(fp," %f", float( u[i]->Get(j) ));
		  fprintf(fp,"\n");
		}

	 }

	 // This one should also be included but that does not seem to work. Strange!
	 //fprintf(fp,"end values\n");

	 fprintf(fp,"\n");
	 
  }
	 
}
//-----------------------------------------------------------------------------
