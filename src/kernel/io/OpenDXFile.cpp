// Copyright (C) 2003 Johan Hoffman and Anders Logg.
// Licensed under the GNU GPL Version 2.

#include <stdio.h>
#include <dolfin/Grid.h>
#include <dolfin/dolfin_log.h>
#include <dolfin/OpenDXFile.h>

using namespace dolfin;

//-­---------------------------------------------------------------------------
OpenDXFile::OpenDXFile(const std::string filename) : GenericFile(filename)
{

}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator>>(Vector& x)
{
  dolfin_error("Cannot read vectors from OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator>>(Matrix& A)
{
  dolfin_error("Cannot read matrices from OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator>>(Grid& grid)
{
  dolfin_error("Cannot read grids from OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator>>(Function& u)
{
  dolfin_error("Cannot read functions from OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Vector& x)
{
  dolfin_error("Cannot save vectors to OpenDX files.");  
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Matrix& A)
{
  dolfin_error("Cannot save matrices to OpenDX files.");
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Grid& grid)
{
  dolfin_info("Saving grid to OpenDX file.");

  int noNodes = grid.noNodes();
  int noCells = grid.noCells();

  // Open file
  FILE* fp = fopen(filename.c_str(), "a");

  // Write nodes
  fprintf(fp, "# A list of all node positions\n");
  fprintf(fp, "object \"nodes\" class array type float rank 1 shape 3 items %d lsb binary data follows\n", noNodes);

  for (NodeIterator n(grid); !n.end(); ++n) {
    
    Point p = n->coord();
    
    float x = (float) p.x;
    float y = (float) p.y;
    float z = (float) p.z;
    
    fwrite(&x, sizeof(float), 1, fp);
    fwrite(&y, sizeof(float), 1, fp);
    fwrite(&z, sizeof(float), 1, fp);
    
  }
  fprintf(fp,"\n\n");

  // Write cells
  fprintf(fp, "# A list of all elements (connections)\n");
  fprintf(fp, "object \"cells\" class array type int rank 1 shape 4 items %d lsb binary data follows\n", noCells);

  for (CellIterator c(grid); !c.end(); ++c) {  
    for (NodeIterator n(c); !n.end(); ++n) {
      int id  = n->id();
      fwrite(&id, sizeof(int), 1, fp);
    }
  }
  fprintf(fp, "\n");
  fprintf(fp, "attribute \"element type\" string \"tetrahedra\"\n");
  fprintf(fp, "attribute \"ref\" string \"positions\"\n");
  fprintf(fp, "\n");  

  // Write data (cell diameter)
  fprintf(fp,"# Cell diameter\n");
  fprintf(fp,"object \"diameter\" class array type float rank 0 items %d lsb binary data follows\n", noCells);
  
  for (CellIterator c(grid); !c.end(); ++c) {
    float value = (float) c->diameter();
    fwrite(&value, sizeof(float), 1, fp);
  }
  fprintf(fp, "\n");
  fprintf(fp, "attribute \"dep\" string \"connections\"\n");
  fprintf(fp, "\n");  

  // Write the grid
  fprintf(fp, "# The grid\n");
  fprintf(fp, "object \"Grid\" class field\n");
  fprintf(fp, "component \"positions\" value \"nodes\"\n");
  fprintf(fp, "component \"connections\" value \"cells\"\n");
  fprintf(fp, "component \"data\" value \"diameter\"\n");

  // Close file
  fclose(fp);
}
//-­---------------------------------------------------------------------------
void OpenDXFile::operator<<(Function& u)
{
  dolfin_error("Cannot save functions to OpenDX files.");
}
//-­---------------------------------------------------------------------------
