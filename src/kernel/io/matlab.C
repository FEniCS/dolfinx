#include "matlab.h"
#include <dolfin/Display.hh>

//----------------------------------------------------------------------------
void matlab_write_header(FILE *fp, DataInfo *datainfo, SysInfo *sysinfo)
{
  fprintf(fp,"%% Output from DOLFIN version %s.\n",DOLFIN_VERSION);
  fprintf(fp,"%%\n");
  fprintf(fp,"%% Saved by %s at %s\n",sysinfo->user,sysinfo->time);
  fprintf(fp,"%% on %s (%s) running %s version %s.\n",
			 sysinfo->host,sysinfo->mach,sysinfo->name,sysinfo->vers);
  fprintf(fp,"%%\n");
  fprintf(fp,"%% %s\n",datainfo->Description());
  fprintf(fp,"%%\n");
  fprintf(fp,"%%    p     coordinates for nodes\n");
  fprintf(fp,"%%    e     edges (temporary fix)\n");
  fprintf(fp,"%%    c     nodes in cells\n");
  fprintf(fp,"%%    t(i)  time values\n");
  fprintf(fp,"%%\n");

  for (int i=0;i<datainfo->Size();i++){
	 if ( datainfo->Dim(i) == 1 )
		fprintf(fp,"%%    %s{i}  %s\n",
				  datainfo->Name(i),datainfo->Label(i));
	 else
		for (int j=0;j<datainfo->Dim(i);j++)
		  fprintf(fp,"%%    %s%d{i} %s, component %d\n",
					 datainfo->Name(i),j+1,datainfo->Label(i),j+1);
  }

  fprintf(fp,"%%\n");
  fprintf(fp,"%% Type the name of this file (without the .m suffix)\n");
  fprintf(fp,"%% in MATLAB and use the pdeplot tools to view the\n");
  fprintf(fp,"%% solution:\n");
  fprintf(fp,"%%\n");
  fprintf(fp,"%%    pdemesh(p,e,c)\n");
  if ( datainfo->Dim(0) == 1 )
	 fprintf(fp,"%%    pdesurf(p,c,%s{1})\n",datainfo->Name(0));
  else
	 fprintf(fp,"%%    pdesurf(p,c,%s1{1})\n",datainfo->Name(0));
  fprintf(fp,"%%    ...\n");
  
  fprintf(fp,"\n");
}
//----------------------------------------------------------------------------
void matlab_write_grid(FILE *fp, Grid *grid)
{
  // Check dimension
  if ( grid->GetCell(0)->GetSize() != 3 )
	 display->Error("Unable to save 3d data to MATLAB format (not implemented).");
  
  Point *p;
  Cell *c;

  // Write nodes
  
  fprintf(fp,"p = [");
  for (int i=0;i<grid->GetNoNodes();i++){
	 p = grid->GetNode(i)->GetCoord();
	 if ( i < (grid->GetNoNodes()-1) )
		fprintf(fp,"%f %f\n",p->x,p->y );
	 else
		fprintf(fp,"%f %f]';\n",p->x,p->y);
  }
  fprintf(fp,"\n");

  // Write cells
  
  fprintf(fp,"c = [");
  for (int i=0;i<grid->GetNoCells();i++){
	 c = grid->GetCell(i);
	 for (int j=0;j<c->GetSize();j++)
		if ( j < (c->GetSize()-1) )
		  fprintf(fp,"%d ",c->GetNode(j)->GetNodeNo()+1);
		else
		  fprintf(fp,"%d",c->GetNode(j)->GetNodeNo()+1);
	 if ( i < (grid->GetNoCells()-1) )
		fprintf(fp," 1\n");
	 else
		fprintf(fp," 1]';\n");
  }
  fprintf(fp,"\n");

  // Write edges (to make the pdeplot routines happy)

  fprintf(fp,"e = [1;2;0;0;0;0;0];\n\n");
  
}
//----------------------------------------------------------------------------
void matlab_write_field(FILE *fp, DataInfo *datainfo, Vector **u, real t,
								int frame, int no_vectors)
{
  int index;
  int offset = 0;
  
  fprintf(fp,"t(%d) = %f;\n",frame+1,t);
  
  // Go through all variables
  for (int i=0;i<datainfo->Size();i++){

	 offset = 0;
	 
	 // Go through all vector dimensions for variable i
	 for (int j=0;j<datainfo->Dim(i);j++){

		if ( no_vectors == 1 )
		  index = 0;
		else
		  index = i;
		
		if ( datainfo->Dim(i) == 1 )
		  fprintf(fp,"%s{%d} = [",datainfo->Name(i),frame+1);
		else
		  fprintf(fp,"%s%d{%d} = [",datainfo->Name(i),j+1,frame+1);

		for (int k=0;k<u[index]->Size();k+=datainfo->Dim(i)){
		  if ( k < (u[index]->Size()-datainfo->Dim(i)) )
			 fprintf(fp,"%1.5e\n",float(u[index]->Get(offset+k)));
		  else
			 fprintf(fp,"%1.5e];\n",float(u[index]->Get(offset+k)));
		}

		if ( no_vectors != 1 )
		  offset += 1;
		
	 }
  }
}
//----------------------------------------------------------------------------
