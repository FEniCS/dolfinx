// Copyright (C) 2004 Harald Svensson.
// Licensed under the GNU GPL Version 2.
//
// Modified by Anders Logg, 2004.
// Modified by Garth N. Wells, 2005.

#include <stdio.h>
#include <dolfin/Mesh.h>
#include <dolfin/Function.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/TecplotFile.h>

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

  // Tetrahedra
  if ( mesh.type() == Mesh::tetrahedrons ) {
  
    // Write header
    fprintf(fp, "TITLE = \"Dolfin Output\"  \n");
    fprintf(fp, "VARIABLES = \"X1\",  \"X2\",  \"X3\"  \n");
    fprintf(fp, "ZONE  N = %8d, E = %8d, F = FEPOINT, ET=Tetrahedron \n", mesh.noNodes(), mesh.noCells());

    // Write node positions
    for (NodeIterator n(mesh); !n.end(); ++n)
    {

      Point   p = n->coord();

      float x1 = (float) p.x;
      float x2 = (float) p.y;
      float x3 = (float) p.z;

      fprintf(fp," %e %ef %e \n",x1, x2, x3);
    }
  
      // Write cell connectivity
     for (CellIterator c(mesh); !c.end(); ++c)
     {
       for (NodeIterator n(c); !n.end(); ++n) fprintf(fp," %8d ",n->id()+1);
       fprintf(fp," \n");
     }  

  }

  // Triangles
  if ( mesh.type() == Mesh::triangles ) {

    // Write header
    fprintf(fp, "TITLE = \"Dolfin Output\"  \n");
    fprintf(fp, "VARIABLES = \"X1\",  \"X2\" \n");
    fprintf(fp, "ZONE  N = %8d, E = %8d, F = FEPOINT, ET=Triangles \n", mesh.noNodes(), mesh.noCells());

      // Write node positions
    for (NodeIterator n(mesh); !n.end(); ++n)
    {

      Point   p = n->coord();

      float x1 = (float) p.x;
      float x2 = (float) p.y;

      fprintf(fp," %e %e  \n",x1, x2);

    }

      // Write cell connectivity
     for (CellIterator c(mesh); !c.end(); ++c)
     {
       for (NodeIterator n(c); !n.end(); ++n) fprintf(fp," %8d ",n->id()+1);
       fprintf(fp," \n");
     }  

  }

  // Close file
  fclose(fp);

}
//-­---------------------------------------------------------------------------
void TecplotFile::operator<<(Function& u)
{
  dolfin_warning("Cannot write functions to Tecplot files.");
}
//-­---------------------------------------------------------------------------
void TecplotFile::operator<<(Function::Vector& u)
{
  // Assume mesh is the same for all components
  Mesh* mesh = &(u(0).mesh());

  // Open file
  FILE *fp = fopen(filename.c_str(), "a");

  // Write mesh the first time
  if ( u(0).number() == 0 )
  {
    // Tetrahedra
    if ( mesh->type() == Mesh::tetrahedrons ) {
  
      // Write header
      fprintf(fp, "TITLE = \"Dolfin output\"  \n");
      fprintf(fp, "VARIABLES = \"X1\",  \"X2\",  \"X3\" ");
      for (int i=0; i<u.size(); ++i) fprintf(fp, " ,\"U%d\"  ", i+1);	  
      fprintf(fp, "\n");	  
	  fprintf(fp, "ZONE T = \"%6d\" N = %8d, E = %8d, F = FEPOINT, ET=Tetrahedron \n", u(0).number(), mesh->noNodes(), mesh->noCells());

      // Write node positions and results
      for (NodeIterator n(mesh); !n.end(); ++n)
      {

        Point   p = n->coord();

        float x1 = (float) p.x;
        float x2 = (float) p.y;
        float x3 = (float) p.z;

        fprintf(fp," %e %e %e  ",x1, x2, x3);
        for (int i=0; i < u.size(); ++i) fprintf(fp,"%e ", u(i)(*n) );
        fprintf(fp,"\n");
      }

      // Write cell connectivity
     for (CellIterator c(mesh); !c.end(); ++c)
     {
       for (NodeIterator n(c); !n.end(); ++n) fprintf(fp," %8d ",n->id()+1);
       fprintf(fp," \n");
     }  

    }

    // Triangles
    if ( mesh->type() == Mesh::triangles ) {
  	  
	  // Write header
      fprintf(fp, "TITLE = \"Dolfin output\"  \n");
      fprintf(fp, "VARIABLES = \"X1\",  \"X2\"");
      for (int i=0; i<u.size(); ++i) fprintf(fp, " ,\"U%d\"  ", i+1);	  
      fprintf(fp, "\n");	  
      fprintf(fp, "ZONE T = \"%6d\"  N = %8d, E = %8d, F = FEPOINT, ET=Triangle \n", u(0).number(), mesh->noNodes(), mesh->noCells());

      // Write nodes and results
      for (NodeIterator n(mesh); !n.end(); ++n)
      {

        Point   p = n->coord();

        float x1 = (float) p.x;
        float x2 = (float) p.y;

        fprintf(fp," %e %e  ",x1, x2);
        for (int i=0; i < u.size(); ++i) fprintf(fp,"%e ", u(i)(*n) );
        fprintf(fp,"\n");
      }

      // Write cell connectivity
     for (CellIterator c(mesh); !c.end(); ++c)
     {
       for (NodeIterator n(c); !n.end(); ++n) fprintf(fp," %8d ",n->id()+1);
       fprintf(fp," \n");
     }  

    }
  }

  // Write data for seccond and subsequent times
  if ( u(0).number() != 0 )
  {
    // Tetrahedra
    if ( mesh->type() == Mesh::tetrahedrons ) {
  
      // Write header
      fprintf(fp, "ZONE T = \"%6d\" N = %8d, E = %8d, F = FEPOINT, ET=Tetrahedron, D=(FECONNECT) \n", u(0).number(), mesh->noNodes(), mesh->noCells());

      // Write node positions and results
      for (NodeIterator n(mesh); !n.end(); ++n)
      {

        Point   p = n->coord();

        float x1 = (float) p.x;
        float x2 = (float) p.y;
        float x3 = (float) p.z;

        fprintf(fp," %e %e %e  ",x1, x2, x3);
        for (int i=0; i < u.size(); ++i) fprintf(fp,"%e ", u(i)(*n) );
        fprintf(fp,"\n");
      }

    }

    // Triangles
    if ( mesh->type() == Mesh::triangles ) {
  
      // Write header
      fprintf(fp, "ZONE T = \"%6d\"  N = %8d, E = %8d, F = FEPOINT, ET=Triangle, D=(FECONNECT) \n", u(0).number(), mesh->noNodes(), mesh->noCells());

      // Write node positions and results
      for (NodeIterator n(mesh); !n.end(); ++n)
      {

        Point   p = n->coord();

        float x1 = (float) p.x;
        float x2 = (float) p.y;

        fprintf(fp," %e %e  ",x1, x2);
        for (int i=0; i < u.size(); ++i) fprintf(fp,"%e ", u(i)(*n) );
        fprintf(fp,"\n");
      }

    }
  }
    
  // Close file
  fclose(fp);
  
  // Increase the number of times we have saved the function
  ++u(0);

  cout << "Saved function " << u(0).name() << " (" << u(0).label()
       << ") to file " << filename << " in Tecplot format." << endl;
}
//-­---------------------------------------------------------------------------
