// Copyright (C) 2004-2005 Garth N. Wells.
// Licensed under the GNU LGPL Version 2.1.
//
// Modified by Anders Logg 2004-2006.
//
// First added:  2004
// Last changed: 2006-10-09

#include <dolfin/Mesh.h>
#include <dolfin/Function.h>
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
  if ( mesh.type().cellType() == CellType::tetrahedron ){
	  fprintf(fp, " X1  X2  X3 \n");
	  fprintf(fp, "ZONE T = \" - \" N = %8u, E = %8u, DATAPACKING = POINT, ZONETYPE=FETETRAHEDRON \n", 
	     mesh.numVertices(), mesh.numCells());
  }
	if ( mesh.type().cellType() == CellType::triangle ){    
	  fprintf(fp, " X1  X2  X3 \n");
    fprintf(fp, "ZONE T = \" - \"  N = %8u, E = %8u,  DATAPACKING = POINT, ZONETYPE=FETRIANGLE \n",   
        mesh.numVertices(), mesh.numCells());
   }

  // Write vertex locations
  for (VertexIterator n(mesh); !n.end(); ++n)
  {
    Point   p = n->point();

    if ( mesh.type().cellType() == CellType::tetrahedron )  fprintf(fp," %e %e %e \n", p.x(), p.y(), p.z());
    if ( mesh.type().cellType() == CellType::triangle )     fprintf(fp," %e %e  ", p.x(), p.y());
    fprintf(fp,"\n");

  }

  // Write cell connectivity
  for (CellIterator c(mesh); !c.end(); ++c)
  {
    for (VertexIterator n(c); !n.end(); ++n) fprintf(fp," %u ",n->index()+1);
    fprintf(fp," \n");
  }  

  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
void TecplotFile::operator<<(Function& u)
{
  FILE *fp = fopen(filename.c_str(), "a");
	
  // Write mesh the first time
  if ( counter == 0 )
  {
	  // Write header
    fprintf(fp, "TITLE = \"Dolfin output\"  \n");
    fprintf(fp, "VARIABLES = ");
    for (uint i = 0; i < u.element().shapedim(); ++i)   fprintf(fp, " X%u  ", i+1);	  
    for (uint i = 0; i < u.vectordim(); ++i)  fprintf(fp, " U%u  ", i+1);	  
    fprintf(fp, "\n");	  

    if ( u.mesh().type().cellType() == CellType::tetrahedron )
	     fprintf(fp, "ZONE T = \"%6u\" N = %8u, E = %8u, DATAPACKING = POINT, ZONETYPE=FETETRAHEDRON \n", 
	         counter+1, u.mesh().numVertices(), u.mesh().numCells());
    if ( u.mesh().type().cellType() == CellType::triangle )
       fprintf(fp, "ZONE T = \"%6u\"  N = %8u, E = %8u, DATAPACKING = POINT, ZONETYPE=FETRIANGLE \n",
            counter+1, u.mesh().numVertices(), u.mesh().numCells());


    // Write vertex locations and results
    for (VertexIterator n(u.mesh()); !n.end(); ++n)
    {
      Point p = n->point();

      if ( u.mesh().type().cellType() == CellType::tetrahedron )  fprintf(fp," %e %e %e \n", p.x(), p.y(), p.z());
      if ( u.mesh().type().cellType() == CellType::triangle )     fprintf(fp," %e %e  ", p.x(), p.y());
      for (uint i=0; i < u.vectordim(); ++i) fprintf(fp,"%e ", u(*n,i) );
      fprintf(fp,"\n");

      }

      // Write cell connectivity
     for (CellIterator c(u.mesh()); !c.end(); ++c)
     {
       for (VertexIterator n(c); !n.end(); ++n) fprintf(fp," %8u ",(n->index())+1);
       fprintf(fp," \n");
     }  

   }

  // Write data for second and subsequent times
  if ( counter != 0 )
  {
      // Write header

    if ( u.mesh().type().cellType() == CellType::tetrahedron )
      fprintf(fp, "ZONE T = \"%6u\" N = %8u, E = %8u,  DATAPACKING = POINT, ZONETYPE=FETETRAHEDRON, VARSHARELIST = ([1-3]=1) CONNECTIVITYSHAREZONE=1 \n", 
	      counter+1, u.mesh().numVertices(), u.mesh().numCells());
    if ( u.mesh().type().cellType() == CellType::triangle )
      fprintf(fp, "ZONE T = \"%6u\"  N = %8u, E = %8u, DATAPACKING = POINT, ZONETYPE=FETRIANGLE, VARSHARELIST = ([1,2]=1) CONNECTIVITYSHAREZONE=1 \n", 
              counter+1, u.mesh().numVertices(), u.mesh().numCells());
    
    // Write vertex locations and results
    for (VertexIterator n(u.mesh()); !n.end(); ++n)
    {
      for (uint i=0; i < u.vectordim(); ++i) fprintf(fp,"%e ", u(*n,i) );
      fprintf(fp,"\n");
    }
  }
    
  // Close file
  fclose(fp);
  
  // Increase the number of times we have saved the function
  counter++;

  cout << "Saved function " << u.name() << " (" << u.label()
       << ") to file " << filename << " in Tecplot format." << endl;
}
//-­---------------------------------------------------------------------------
