#include "opendx.h"
#include <Display.hh>

//-----------------------------------------------------------------------------
void opendx_write_header(FILE *fp, DataInfo *datainfo, SysInfo *sysinfo)
{
  fprintf(fp,"# Output from DOLFIN version %s.\n",DOLFIN_VERSION);
  fprintf(fp,"# Format is intended for use with OpenDX (Data Explorer).\n");
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
void opendx_write_grid(FILE *fp, Grid *grid)
{
  // Check dimension
  if ( grid->GetCell(0)->GetSize() != 4 )
	 display->Error("Unable to save 2d data to OpenDX format (not implemented).");
  
  int noNodes = grid->GetNoNodes();
  int noElems = grid->GetNoCells();
  int i;
  float x,y,z;
  int n1, n2, n3, n4;
  
  // Write nodes

  Point *p;
  Cell *c;
  
  fprintf(fp,"# A list of all node positions\n");
  fprintf(fp,"object 1 class array type float rank 1 shape 3 items %d lsb binary data follows\n",
                         noNodes);
  
  for(i=0;i<noNodes;i++){

	 p = grid->GetNode(i)->GetCoord();
	 
	 x = float(p->x);
	 y = float(p->y);
	 z = float(p->z);
	 
	 fwrite(&x,sizeof(float),1,fp);
	 fwrite(&y,sizeof(float),1,fp);
	 fwrite(&z,sizeof(float),1,fp);
         
  }
  fprintf(fp,"\n\n");

  // Write cells
  
  fprintf(fp,"# A list of all elements (connections)\n");
  fprintf(fp,"object 2 class array type int rank 1 shape 4 items %d lsb binary data follows\n",noElems);
  
  for(i=0;i<noElems;i++){

	 c = grid->GetCell(i);
	 
	 n1 = c->GetNode(0)->GetNodeNo();
	 n2 = c->GetNode(1)->GetNodeNo();
	 n3 = c->GetNode(2)->GetNodeNo();
	 n4 = c->GetNode(3)->GetNodeNo();

	 fwrite(&n1,sizeof(int),1,fp);
	 fwrite(&n2,sizeof(int),1,fp);
	 fwrite(&n3,sizeof(int),1,fp);
	 fwrite(&n4,sizeof(int),1,fp);
	 
  }
  fprintf(fp,"\n");
  fprintf(fp,"attribute \"element type\" string \"tetrahedra\"\n");
  fprintf(fp,"attribute \"ref\" string \"positions\"\n");
  fprintf(fp,"\n\n");  

  // Increase the object counter
  //iObject += 2;
  
}
//-----------------------------------------------------------------------------
void opendx_write_field(FILE *fp, DataInfo *datainfo, Grid *grid, int frame,
								Vector **u, int no_vectors)
{
  float value;

  int no_nodes = grid->GetNoNodes();
  int offset   = 0;
  int object   = 3 + 2*datainfo->Size()*frame;

  // Compute total data size
  int datasize = 0;
  for (int i=0;i<datainfo->Size();i++)
	 datasize += datainfo->Dim(i);

  // Go trough all variables
  for (int i=0;i<datainfo->Size();i++){
	 
	 // Write header for object
	 fprintf(fp,"# Values for [%s] at nodal points, frame %d\n",datainfo->Label(i),frame+1);
	 fprintf(fp,"object %d class array type float rank %d shape %d items %d lsb binary data follows\n",
				object,1,datainfo->Dim(i),no_nodes);

	 if ( no_vectors == 1 ){
	 
		// Go through all (nodal) values
		for (int j=0;j<no_nodes;j++){
		  
		  // Go through all components of the variable
		  for (int k=0;k<datainfo->Dim(i);k++){
			 
			 // Get the value
			 value = float( u[0]->Get(j*datasize+offset+k) );
			 
			 // Write the value
			 fwrite(&value,sizeof(float),1,fp);
			 
		  }
		  
		}
		
		// Increase offset
		offset += datainfo->Dim(i);

	 }
	 else{

		// Go through all (nodal) values
		for (int j=0;j<u[i]->Size();j++){
		  
		  // Get the value
		  value = float( u[i]->Get(j) );
		  
		  // Write the value
		  fwrite(&value,sizeof(float),1,fp);
		  
		}

	 }
	 
	 fprintf(fp,"\n");
	 fprintf(fp,"attribute \"dep\" string \"positions\"\n\n");
	 
	 // Increase the object counter
	 object += 1;
	 
	 // Write field
	 fprintf(fp,"# Field for [%s], frame %d\n",datainfo->Label(i),frame);
	 fprintf(fp,"object %d class field\n",object);
	 fprintf(fp,"component \"positions\" value 1\n");
	 fprintf(fp,"component \"connections\" value 2\n");
	 fprintf(fp,"component \"data\" value %d\n\n",object-1);
	 
	 // Increase the object counter
	 object += 1;	 
	 	 
  }

}
//-----------------------------------------------------------------------------
void opendx_remove_series(FILE *fp, DataInfo *datainfo)
{
  // Remove the previous series (the new one will be placed at the end).
  // This makes sure that we have a valid dx-file even if we kill the
  // program after a few frames.
  
  char c;
  int removed = 0;

  // Step to the end of the file
  fseek(fp,0L,SEEK_END);
  
  while ( true ){

	 // One step back and read a character
	 fseek(fp,-1,SEEK_CUR);
	 c = fgetc(fp);

	 // Check if we reached a '#'
	 // Warning: if someone puts a # inside a comment we are in trouble
	 if ( c == '#' )
		removed += 1;

	 // Check if we removed all series
	 if ( removed == datainfo->Size() ){
		fflush(fp);
		break;
	 }
	 
	 // Step back again to before the character
	 fseek(fp,-1,SEEK_CUR);

  }
  
}
//-----------------------------------------------------------------------------
void opendx_write_series(FILE *fp, DataInfo *datainfo, int frame, Vector *t)
{
  // Write the time series

  for (int i=0;i<datainfo->Size();i++){
	 
	 fprintf(fp,"# Time series for [%s]\n",datainfo->Label(i));
	 fprintf(fp,"object \"%s\" class series\n",datainfo->Name(i));
	 
	 for (int j=0;j<=frame;j++)
		fprintf(fp,"member %d value %d position %f\n",
				  j,
				  3 + 2*datainfo->Size()*j + 2*i + 1,
				  t->Get(j));
	 
	 fprintf(fp,"\n");
	 
  }
  
}
//-----------------------------------------------------------------------------
