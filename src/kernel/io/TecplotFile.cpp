// Copyright (C) 2004-2005 Garth N. Wells.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg 2004-2005.
//
// First added:  2004
// Last changed: 2005

#include <dolfin/Mesh.h>
#include <dolfin/TecplotFile.h>
#include <dolfin/FiniteElement.h>

using namespace dolfin;

//-­---------------------------------------------------------------------------
TecplotFile::TecplotFile(const std::string filename) : GenericFile(filename)
{
  type = "TECPLOT";
}
//-­---------------------------------------------------------------------------
TecplotFile::~TecplotFile()
{
  // Do nothing
}
//-­---------------------------------------------------------------------------
void TecplotFile::operator<<(Mesh& mesh)
{

  dolfin_info("Saving mesh to Tecplot file.");

  // Open file
  FILE* fp = fopen(filename.c_str(), "a");

	// Write header
  fprintf(fp, "TITLE = \"Dolfin output\"  \n");
  fprintf(fp, "VARIABLES = ");
  if ( mesh.type() == Mesh::tetrahedra ){
	  fprintf(fp, " X1  X2  X3 \n");
	  fprintf(fp, "ZONE T = \" - \" N = %8d, E = %8d, DATAPACKING = POINT, ZONETYPE=FETETRAHEDRON \n", mesh.noNodes(), mesh.noCells());
  }
	if ( mesh.type() == Mesh::triangles ){    
	  fprintf(fp, " X1  X2  X3 \n");
    fprintf(fp, "ZONE T = \" - \"  N = %8d, E = %8d,  DATAPACKING = POINT, ZONETYPE=FETRIANGLE \n",   mesh.noNodes(), mesh.noCells());
   }

  // Write node locations
  for (NodeIterator n(mesh); !n.end(); ++n)
  {
    Point   p = n->coord();

    if ( mesh.type() == Mesh::tetrahedra )  fprintf(fp," %e %e %e \n",p.x, p.y, p.z);
    if ( mesh.type() == Mesh::triangles )     fprintf(fp," %e %e  ",p.x, p.y);
    fprintf(fp,"\n");

  }

  // Write cell connectivity
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (NodeIterator n(c); !n.end(); ++n) fprintf(fp," %8d ",n->id()+1);
    fprintf(fp," \n");
  }  

  // Close file
  fclose(fp);

}
//-­---------------------------------------------------------------------------
void TecplotFile::operator<<(Function& u)
{

  const FiniteElement& element = u.element();
	
	FILE *fp = fopen(filename.c_str(), "a");
    
  uint ShapeDim  = element.shapedim();
  uint VectorDim = 0;

  if ( element.rank() == 0 )
  {
    VectorDim = 1;
  }
  else if ( element.rank() == 1 )
  {
    VectorDim = element.tensordim(0);
  }
	
  // Write mesh the first time
  if ( u.number() == 0 )
  {
	  // Write header
    fprintf(fp, "TITLE = \"Dolfin output\"  \n");
    fprintf(fp, "VARIABLES = ");
    for (uint i=0; i<ShapeDim; ++i)   fprintf(fp, " X%d  ", i+1);	  
    for (uint i=0; i<VectorDim; ++i)  fprintf(fp, " U%d  ", i+1);	  
    fprintf(fp, "\n");	  
    if ( u.mesh().type() == Mesh::tetrahedra )
	     fprintf(fp, "ZONE T = \"%6d\" N = %8d, E = %8d, DATAPACKING = POINT, ZONETYPE=FETETRAHEDRON \n", u.number()+1, u.mesh().noNodes(), u.mesh().noCells());
    if ( u.mesh().type() == Mesh::triangles )
       fprintf(fp, "ZONE T = \"%6d\"  N = %8d, E = %8d, DATAPACKING = POINT, ZONETYPE=FETRIANGLE \n",   u.number()+1, u.mesh().noNodes(), u.mesh().noCells());


    // Write node locations and results
    for (NodeIterator n(u.mesh()); !n.end(); ++n)
    {
      Point p = n->coord();

      if ( u.mesh().type() == Mesh::tetrahedra )  fprintf(fp," %e %e %e \n", p.x, p.y, p.z);
      if ( u.mesh().type() == Mesh::triangles )     fprintf(fp," %e %e  ", p.x, p.y);
      for (uint i=0; i < VectorDim; ++i) fprintf(fp,"%e ", u(*n,i) );
      fprintf(fp,"\n");

      }

      // Write cell connectivity
     for (CellIterator c(u.mesh()); !c.end(); ++c)
     {
       for (NodeIterator n(c); !n.end(); ++n) fprintf(fp," %8d ",n->id()+1);
       fprintf(fp," \n");
     }  

   }
	  

  // Write data for seccond and subsequent times
  if ( u.number() != 0 )
  {
      // Write header
	  if ( u.mesh().type() == Mesh::tetrahedra )
      	   fprintf(fp, "ZONE T = \"%6d\" N = %8d, E = %8d,  DATAPACKING = POINT, ZONETYPE=FETETRAHEDRON, VARSHARELIST = ([1-3]=1) CONNECTIVITYSHAREZONE=1 \n", u.number()+1, u.mesh().noNodes(), u.mesh().noCells());
     if ( u.mesh().type() == Mesh::triangles )
           fprintf(fp, "ZONE T = \"%6d\"  N = %8d, E = %8d, DATAPACKING = POINT, ZONETYPE=FETRIANGLE, VARSHARELIST = ([1,2]=1) CONNECTIVITYSHAREZONE=1 \n", u.number()+1, u.mesh().noNodes(), u.mesh().noCells());


      // Write node locations and results
      for (NodeIterator n(u.mesh()); !n.end(); ++n)
      {

        for (uint i=0; i < VectorDim; ++i) fprintf(fp,"%e ", u(*n,i) );
        fprintf(fp,"\n");

      }
  }
    
  // Close file
  fclose(fp);
  
  // Increase the number of times we have saved the function
  ++u;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in Tecplot format." << endl;



}
//-­---------------------------------------------------------------------------
